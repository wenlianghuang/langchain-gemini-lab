import { useState, useRef, useEffect } from 'react'
import './App.css'

// 使用相对路径，通过 Vite 代理转发到后端
// 使用 invoke 端点，非流式响应（更简单可靠）
const API_INVOKE_URL = '/agent/invoke'
const THREAD_ID = 'web-user-demo'

function App() {
    const [messages, setMessages] = useState([
        {
            type: 'ai',
            content: '你好！我是具備股票、RAG 和網路搜尋能力的 Super Agent。請問有什麼可以為您服務的？'
        }
    ])
    const [input, setInput] = useState('')
    const [isStreaming, setIsStreaming] = useState(false)
    const [status, setStatus] = useState('')
    const messagesEndRef = useRef(null)
    const messageHistoryRef = useRef([
        { type: "system", content: "你是一個智能助手，可以使用工具來回答問題，請保持簡潔。" }
    ])

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [messages])

    const extractContent = (msg) => {
        if (typeof msg.content === "string") {
            return msg.content
        } else if (Array.isArray(msg.content)) {
            return msg.content.map(item =>
                typeof item === "string" ? item : item.text || ""
            ).join("")
        }
        return ""
    }

    const handleSend = async () => {
        if (!input.trim() || isStreaming) return

        const userText = input.trim()
        setInput('')
        setIsStreaming(true)
        setStatus('Agent 思考中...')

        // 添加用户消息到 UI
        const userMessage = { type: 'user', content: userText }
        setMessages(prev => [...prev, userMessage])

        // 添加用户消息到历史记录（使用 LangChain 格式）
        messageHistoryRef.current.push({ type: "human", content: userText })

        // 创建空的 AI 消息占位符
        const aiMessageId = Date.now()
        setMessages(prev => [...prev, { type: 'ai', content: '思考中...', id: aiMessageId }])

        try {
            // 构建符合 LangServe 格式的请求体
            const body = {
                input: {
                    messages: messageHistoryRef.current.map(msg => {
                        // 确保消息格式符合 LangChain 的消息类型
                        if (msg.type === "system") {
                            return { type: "system", content: msg.content }
                        } else if (msg.type === "human") {
                            return { type: "human", content: msg.content }
                        } else if (msg.type === "ai") {
                            return { type: "ai", content: msg.content }
                        }
                        return msg
                    })
                },
                config: {
                    configurable: {
                        thread_id: THREAD_ID
                    }
                }
            }

            console.log("📤 发送请求:", JSON.stringify(body, null, 2))

            // 使用 invoke 端点，等待完整响应
            const response = await fetch(API_INVOKE_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            })

            if (!response.ok) {
                const errorText = await response.text()
                console.error("❌ 服务器错误:", response.status, errorText)
                throw new Error(`服务器错误 (${response.status}): ${errorText}`)
            }

            // 解析 JSON 响应
            const result = await response.json()
            console.log("📥 收到完整响应:", result)

            // LangServe invoke 端点的响应格式通常是: { output: { messages: [...] } }
            let messagesArray = null
            if (result.output?.messages) {
                messagesArray = result.output.messages
                console.log("✅ 使用路径: result.output.messages")
            } else if (result.messages) {
                messagesArray = result.messages
                console.log("✅ 使用路径: result.messages")
            } else if (result.data?.output?.messages) {
                messagesArray = result.data.output.messages
                console.log("✅ 使用路径: result.data.output.messages")
            }

            if (!messagesArray || !Array.isArray(messagesArray)) {
                console.error("❌ 响应格式不正确，完整响应:", JSON.stringify(result, null, 2))
                throw new Error("服务器返回格式不正确")
            }

            console.log(`📋 收到 ${messagesArray.length} 条消息`)

            // 打印所有消息的详细信息（用于调试）
            messagesArray.forEach((msg, idx) => {
                console.log(`   消息 ${idx}: type=${msg.type}, hasContent=${!!msg.content}, hasToolCalls=${!!(msg.tool_calls?.length)}`)
            })

            // 从后往前找，找到最后一条有 content 的 AI 消息（且没有 tool_calls）
            let finalResponse = ""
            for (let i = messagesArray.length - 1; i >= 0; i--) {
                const msg = messagesArray[i]

                // 跳过工具调用中的消息，只取最终回答
                if (msg.type === "ai" && msg.content &&
                    (!msg.tool_calls || msg.tool_calls.length === 0)) {

                    finalResponse = extractContent(msg)
                    console.log(`✅ 找到最终回答: ${finalResponse.substring(0, 100)}...`)
                    break
                }
            }

            if (finalResponse) {
                // 更新 UI
                setMessages(prev => prev.map(m =>
                    m.id === aiMessageId
                        ? { ...m, content: finalResponse }
                        : m
                ))
                // 更新历史记录
                messageHistoryRef.current.push({ type: "ai", content: finalResponse })
                console.log("✅ 响应处理完成，内容已更新到 UI")
            } else {
                console.warn("⚠️ 未找到有效响应，可能还在工具调用中")
                setMessages(prev => prev.map(m =>
                    m.id === aiMessageId
                        ? { ...m, content: "抱歉，未能获取到有效响应。请重试。" }
                        : m
                ))
            }

        } catch (error) {
            console.error("请求错误:", error)
            setStatus(`错误: ${error.message}`)
            setMessages(prev => prev.map(m =>
                m.id === aiMessageId
                    ? { ...m, content: `[錯誤] 無法取得回應：${error.message}\n\n請確認後端 Server 是否運行在 http://localhost:8000` }
                    : m
            ))
        } finally {
            setIsStreaming(false)
            setStatus('')
        }
    }

    return (
        <div className="app">
            <div className="container">
                <h1>LangGraph Agent Chat</h1>
                <div className="chat-window">
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`message ${msg.type}-msg`}>
                            {msg.content}
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>
                <div className="input-container">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                        placeholder="輸入您的問題..."
                        disabled={isStreaming}
                    />
                    <button onClick={handleSend} disabled={isStreaming}>
                        發送
                    </button>
                </div>
                {status && <p className="status-message">{status}</p>}
            </div>
        </div>
    )
}

export default App

