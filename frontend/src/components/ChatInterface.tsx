'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { SourceNode } from '@/lib/api';
import { useRAGWebSocket, SourceItem } from '@/lib/useWebSocket';
import { createConversation, addMessage as saveMessage, Message as ConvMessage } from '@/lib/conversations';
import SourcesPanel from '@/components/SourcesPanel';
import StreamProgress from '@/components/StreamProgress';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: SourceNode[];
    isLoading?: boolean;
    isStreaming?: boolean;
}

interface ChatInterfaceProps {
    conversationId?: string | null;
    initialMessages?: ConvMessage[];
    onConversationCreated?: (id: string) => void;
}

export default function ChatInterface({
    conversationId,
    initialMessages,
    onConversationCreated
}: ChatInterfaceProps) {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [selectedSources, setSelectedSources] = useState<SourceNode[] | null>(null);
    const [showSources, setShowSources] = useState(false);
    const [isLoaded, setIsLoaded] = useState(false);
    const [showProgress, setShowProgress] = useState(true);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const currentMessageIdRef = useRef<string | null>(null);
    const currentConversationIdRef = useRef<string | null>(conversationId || null);

    // WebSocket hook
    const { streamState, sendQuery, isConnected, resetState } = useRAGWebSocket();

    useEffect(() => {
        setIsLoaded(true);
    }, []);

    // Load initial messages when conversation changes
    useEffect(() => {
        // Skip reset if we're actively streaming - this happens when we just created a new conversation
        // as part of submitting a query, and we don't want to interrupt the ongoing stream
        const isActivelyStreaming = currentMessageIdRef.current !== null;

        console.log('[Chat] Conversation effect running:', {
            conversationId,
            isActivelyStreaming,
            currentMessageId: currentMessageIdRef.current,
            initialMessagesCount: initialMessages?.length ?? 0
        });

        if (!isActivelyStreaming) {
            // Only reset when switching to a DIFFERENT conversation (user clicked on sidebar)
            console.log('[Chat] Resetting state (not streaming)');
            resetState();
            currentMessageIdRef.current = null;

            // Only load initial messages when NOT actively streaming
            if (initialMessages && initialMessages.length > 0) {
                // Convert to UI message format
                const uiMessages: Message[] = initialMessages.map(msg => ({
                    id: msg.id,
                    role: msg.role,
                    content: msg.content,
                    sources: msg.sources as SourceNode[] | undefined,
                    isLoading: false,
                    isStreaming: false,
                }));
                setMessages(uiMessages);
            } else {
                setMessages([]);
            }
        } else {
            console.log('[Chat] Skipping reset and message load (actively streaming)');
        }

        currentConversationIdRef.current = conversationId || null;
    }, [conversationId, initialMessages, resetState]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, streamState.response]);

    // Update message when streaming response changes
    useEffect(() => {
        console.log('[Chat] Streaming update effect:', {
            hasMessageId: !!currentMessageIdRef.current,
            responseLength: streamState.response.length,
            isStreaming: streamState.isStreaming
        });
        if (currentMessageIdRef.current && streamState.response) {
            setMessages(prev => prev.map(msg =>
                msg.id === currentMessageIdRef.current
                    ? { ...msg, content: streamState.response, isLoading: false, isStreaming: streamState.isStreaming }
                    : msg
            ));
        }
    }, [streamState.response, streamState.isStreaming]);

    // Update message with sources when done and save to DB
    useEffect(() => {
        console.log('[Chat] Done effect:', {
            isStreaming: streamState.isStreaming,
            hasMessageId: !!currentMessageIdRef.current,
            messageId: currentMessageIdRef.current,
            responseLength: streamState.response.length,
            sourcesCount: streamState.sources.length
        });
        // When streaming ends (isStreaming goes from true to false)
        if (!streamState.isStreaming && currentMessageIdRef.current && streamState.response) {
            console.log('[Chat] Processing done - updating message UI');
            const sources: SourceNode[] = streamState.sources.map((s: SourceItem) => ({
                content: s.content,
                score: s.score,
                type: s.type,
                metadata: s.metadata,
            }));

            setMessages(prev => prev.map(msg =>
                msg.id === currentMessageIdRef.current
                    ? {
                        ...msg,
                        content: streamState.response,
                        sources: sources.length > 0 ? sources : msg.sources,
                        isLoading: false,
                        isStreaming: false
                    }
                    : msg
            ));

            // Note: Assistant message is saved by the backend (websocket.py)
            // to avoid duplicate saves

            currentMessageIdRef.current = null;
        }
    }, [streamState.isStreaming, streamState.sources, streamState.response]);

    // Handle errors
    useEffect(() => {
        if (streamState.error && currentMessageIdRef.current) {
            setMessages(prev => prev.map(msg =>
                msg.id === currentMessageIdRef.current
                    ? { ...msg, content: `Error: ${streamState.error}`, isLoading: false, isStreaming: false }
                    : msg
            ));
            currentMessageIdRef.current = null;
        }
    }, [streamState.error]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || streamState.isStreaming) return;

        const query = input.trim();
        setInput('');

        // Generate IDs upfront
        const userMessageId = Date.now().toString();
        const assistantId = (Date.now() + 1).toString();

        // Set the message ID ref FIRST - this is checked by the conversation change effect
        // to prevent resetting state when we create a new conversation
        currentMessageIdRef.current = assistantId;
        console.log('[Chat] handleSubmit: Set currentMessageIdRef to:', assistantId);

        const userMessage: Message = {
            id: userMessageId,
            role: 'user',
            content: query,
        };

        // Add messages to UI immediately
        setMessages(prev => [
            ...prev,
            userMessage,
            { id: assistantId, role: 'assistant', content: '', isLoading: true, isStreaming: true }
        ]);

        // Create conversation if needed
        let convId = currentConversationIdRef.current;
        if (!convId) {
            try {
                const newConv = await createConversation(query.slice(0, 50));
                convId = newConv.id;
                currentConversationIdRef.current = convId;
                // Now when this triggers the parent update and useEffect runs,
                // currentMessageIdRef.current will already be set
                onConversationCreated?.(convId);
            } catch (err) {
                console.error('Failed to create conversation:', err);
            }
        }

        // Save user message to Supabase
        if (convId) {
            saveMessage(convId, 'user', query).catch(console.error);
        }

        // Send query via WebSocket
        sendQuery(query, {
            includeSources: true,
            detailed: true,
            conversationId: convId || undefined
        });
    };

    const quickPrompts = [
        'What is in this document?',
        'Summarize the key points',
        'Find related images',
    ];

    // Get current status message from phases
    const getCurrentStatus = () => {
        if (streamState.currentPhase) {
            const phaseMessages: Record<string, string> = {
                embedding: '🔮 Creating embeddings...',
                search: '🔍 Searching documents...',
                reranking: '⚖️ Reranking results...',
                fusion: '🔀 Multi-modal fusion...',
                generation: '✨ Generating response...',
            };
            return phaseMessages[streamState.currentPhase] || 'Processing...';
        }
        return streamState.isStreaming ? 'Connecting...' : '';
    };

    return (
        <div className="flex min-h-screen pt-20">
            {/* Main Content */}
            <div className="flex-1 flex flex-col max-w-4xl mx-auto px-6 w-full">

                {/* Empty State - Hero */}
                {messages.length === 0 && (
                    <div className={`flex-1 flex flex-col items-center justify-center text-center py-20 transition-all duration-1000 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'}`}>

                        {/* Badge */}
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-[var(--border-subtle)] text-xs text-[var(--text-muted)] mb-8">
                            <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-[var(--accent-gold)]'} animate-pulse`} />
                            {isConnected ? 'WebSocket Connected' : 'Multi-Modal RAG'}
                        </div>

                        {/* Hero Text */}
                        <h1 className="font-serif text-5xl md:text-6xl lg:text-7xl font-medium tracking-tight mb-6 leading-[1.1]">
                            <span className="text-[var(--text-primary)]">Unlock your</span>
                            <br />
                            <span className="font-serif italic text-gradient">document</span>
                            <span className="text-[var(--text-primary)]"> insights</span>
                        </h1>

                        <p className="text-[var(--text-secondary)] text-lg max-w-md mb-12 leading-relaxed">
                            Ask questions about your documents with AI-powered multi-modal retrieval
                        </p>

                        {/* Quick Prompts */}
                        <div className="flex flex-wrap justify-center gap-3">
                            {quickPrompts.map((prompt) => (
                                <button
                                    key={prompt}
                                    onClick={() => setInput(prompt)}
                                    className="glass-card glass-card-hover px-5 py-3 text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
                                >
                                    {prompt}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Messages */}
                {messages.length > 0 && (
                    <div className="flex-1 overflow-y-auto py-8 space-y-6">
                        {messages.map((message, index) => (
                            <div key={message.id}>
                                <div
                                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in-up`}
                                    style={{ animationDelay: `${index * 0.05}s` }}
                                >
                                    <div
                                        className={`max-w-[85%] ${message.role === 'user'
                                            ? 'bg-gradient-to-r from-[var(--accent-gold)] to-[#c9976a] text-[var(--bg-primary)]'
                                            : 'glass-card'
                                            } rounded-2xl px-5 py-4`}
                                    >
                                        {message.isLoading && !message.content ? (
                                            <div className="flex items-center gap-3">
                                                <div className="flex gap-1">
                                                    <div className="typing-dot" />
                                                    <div className="typing-dot" />
                                                    <div className="typing-dot" />
                                                </div>
                                                <span className="text-[var(--text-muted)] text-sm">{getCurrentStatus()}</span>
                                            </div>
                                        ) : (
                                            <>
                                                <p className="whitespace-pre-wrap leading-relaxed text-[15px]">
                                                    {message.content}
                                                    {message.isStreaming && <span className="inline-block w-2 h-4 ml-1 bg-[var(--accent-gold)] animate-pulse" />}
                                                </p>
                                                {message.sources && message.sources.length > 0 && !message.isStreaming && (
                                                    <button
                                                        onClick={() => {
                                                            setSelectedSources(message.sources!);
                                                            setShowSources(true);
                                                        }}
                                                        className="mt-4 text-sm text-[var(--accent-gold)] hover:underline flex items-center gap-2 transition-colors"
                                                    >
                                                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                                        </svg>
                                                        View {message.sources.length} sources
                                                    </button>
                                                )}
                                            </>
                                        )}
                                    </div>
                                </div>

                                {/* Show StreamProgress for streaming assistant messages */}
                                {message.role === 'assistant' && message.isStreaming && showProgress && (
                                    <div className="mt-3 ml-0">
                                        <StreamProgress
                                            currentPhase={streamState.currentPhase}
                                            phases={streamState.phases}
                                            chunksFound={streamState.chunksFound}
                                            isStreaming={streamState.isStreaming}
                                        />
                                    </div>
                                )}
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                )}

                {/* Input Area */}
                <div className="sticky bottom-0 py-6 bg-gradient-to-t from-[var(--bg-primary)] via-[var(--bg-primary)] to-transparent">
                    <form onSubmit={handleSubmit}>
                        <div className="flex gap-3 items-center">
                            <input
                                type="text"
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                placeholder="Ask about your documents..."
                                className="input-elegant flex-1"
                                disabled={streamState.isStreaming}
                            />
                            <button
                                type="button"
                                onClick={() => setShowProgress(!showProgress)}
                                className={`p-3 rounded-xl border transition-colors ${showProgress
                                    ? 'border-[var(--accent-gold)] text-[var(--accent-gold)]'
                                    : 'border-[var(--border-subtle)] text-[var(--text-muted)]'
                                    }`}
                                title={showProgress ? 'Hide pipeline progress' : 'Show pipeline progress'}
                            >
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                                </svg>
                            </button>
                            <button
                                type="submit"
                                disabled={streamState.isStreaming || !input.trim()}
                                className="btn-primary flex items-center gap-2 px-6"
                            >
                                {streamState.isStreaming ? (
                                    <div className="spinner" />
                                ) : (
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
                                    </svg>
                                )}
                                <span className="hidden sm:inline">Send</span>
                            </button>
                        </div>
                    </form>

                    {/* Connection Status & Duration */}
                    <div className="mt-3 flex items-center justify-between text-xs text-[var(--text-muted)]">
                        <div className="flex items-center gap-2">
                            {streamState.isStreaming && (
                                <>
                                    <div className="spinner" style={{ width: 12, height: 12 }} />
                                    {getCurrentStatus()}
                                </>
                            )}
                        </div>
                        {streamState.totalDuration && !streamState.isStreaming && (
                            <span className="text-[var(--text-muted)]">
                                Completed in {(streamState.totalDuration / 1000).toFixed(2)}s
                            </span>
                        )}
                    </div>
                </div>
            </div>

            {/* Sources Panel */}
            <SourcesPanel
                sources={selectedSources}
                isOpen={showSources}
                onClose={() => setShowSources(false)}
            />
        </div>
    );
}
