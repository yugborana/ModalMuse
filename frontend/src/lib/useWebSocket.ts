// WebSocket hook for real-time RAG streaming
// lib/useWebSocket.ts

import { useState, useCallback, useRef, useEffect } from 'react';

const WS_BASE = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
    .replace('http://', 'ws://')
    .replace('https://', 'wss://');

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

export interface ChunkPreview {
    id: string;
    preview: string;
    score: number;
    type: 'text' | 'image';
    metadata?: Record<string, unknown>;
}

export interface PhaseEvent {
    type: 'phase';
    phase: 'embedding' | 'search' | 'reranking' | 'fusion' | 'generation';
    status: 'started' | 'completed';
    message: string;
    duration_ms?: number;
}

export interface ChunksFoundEvent {
    type: 'chunks_found';
    text_count: number;
    image_count: number;
    duration_ms: number;
    message: string;
    chunks: ChunkPreview[];
}

export interface GenerationEvent {
    type: 'generation';
    chunk: string;
}

export interface SourceItem {
    content: string;
    score: number;
    type: 'text' | 'image';
    metadata: Record<string, unknown>;
}

export interface SourcesEvent {
    type: 'sources';
    sources: SourceItem[];
    total_duration_ms?: number;
}

export interface DoneEvent {
    type: 'done';
    total_duration_ms?: number;
}

export interface ErrorEvent {
    type: 'error';
    message: string;
}

export type StreamEvent =
    | PhaseEvent
    | ChunksFoundEvent
    | GenerationEvent
    | SourcesEvent
    | DoneEvent
    | ErrorEvent;

export interface StreamState {
    isStreaming: boolean;
    currentPhase: string | null;
    phases: PhaseEvent[];
    chunksFound: ChunksFoundEvent | null;
    response: string;
    sources: SourceItem[];
    error: string | null;
    totalDuration: number | null;
}

// ═══════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════

export function useRAGWebSocket() {
    const wsRef = useRef<WebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);

    const [streamState, setStreamState] = useState<StreamState>({
        isStreaming: false,
        currentPhase: null,
        phases: [],
        chunksFound: null,
        response: '',
        sources: [],
        error: null,
        totalDuration: null,
    });

    // Connect to WebSocket
    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) return;

        const ws = new WebSocket(`${WS_BASE}/ws/query`);

        ws.onopen = () => {
            console.log('[WS] Connected');
            setIsConnected(true);
        };

        ws.onclose = () => {
            console.log('[WS] Disconnected');
            setIsConnected(false);
            wsRef.current = null;
        };

        ws.onerror = (error) => {
            console.error('[WS] Error:', error);
            setStreamState(prev => ({ ...prev, error: 'WebSocket connection error' }));
        };

        ws.onmessage = (event) => {
            try {
                const data: StreamEvent = JSON.parse(event.data);
                handleEvent(data);
            } catch (e) {
                console.error('[WS] Parse error:', e);
            }
        };

        wsRef.current = ws;
    }, []);

    // Handle incoming events
    const handleEvent = useCallback((event: StreamEvent) => {
        console.log('[WS] Received event:', event.type, event);

        switch (event.type) {
            case 'phase':
                setStreamState(prev => ({
                    ...prev,
                    currentPhase: event.status === 'started' ? event.phase : prev.currentPhase,
                    phases: event.status === 'completed'
                        ? [...prev.phases, event]
                        : prev.phases,
                }));
                break;

            case 'chunks_found':
                setStreamState(prev => ({
                    ...prev,
                    chunksFound: event,
                }));
                break;

            case 'generation':
                setStreamState(prev => ({
                    ...prev,
                    response: prev.response + event.chunk,
                }));
                break;

            case 'sources':
                setStreamState(prev => ({
                    ...prev,
                    sources: event.sources,
                    totalDuration: event.total_duration_ms || null,
                }));
                break;

            case 'done':
                console.log('[WS] Stream complete, setting isStreaming to false');
                setStreamState(prev => ({
                    ...prev,
                    isStreaming: false,
                    currentPhase: null,
                    totalDuration: event.total_duration_ms || prev.totalDuration,
                }));
                break;

            case 'error':
                console.error('[WS] Error event:', event.message);
                setStreamState(prev => ({
                    ...prev,
                    isStreaming: false,
                    error: event.message,
                }));
                break;
        }
    }, []);

    // Send query
    const sendQuery = useCallback((query: string, options?: {
        includeSources?: boolean;
        detailed?: boolean;
        conversationId?: string;
    }) => {
        // Reset state
        setStreamState({
            isStreaming: true,
            currentPhase: null,
            phases: [],
            chunksFound: null,
            response: '',
            sources: [],
            error: null,
            totalDuration: null,
        });

        // Connect if needed
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
            console.log('[WS] Creating new WebSocket connection');
            const ws = new WebSocket(`${WS_BASE}/ws/query`);

            ws.onopen = () => {
                console.log('[WS] Connected, sending query:', query.slice(0, 50));
                setIsConnected(true);
                ws.send(JSON.stringify({
                    query,
                    include_sources: options?.includeSources ?? true,
                    detailed: options?.detailed ?? true,
                    conversation_id: options?.conversationId,
                }));
            };

            ws.onmessage = (event) => {
                try {
                    const data: StreamEvent = JSON.parse(event.data);
                    handleEvent(data);
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };

            ws.onclose = (event) => {
                console.log('[WS] Connection closed:', event.code, event.reason);
                setIsConnected(false);
                wsRef.current = null;
            };

            ws.onerror = () => {
                setStreamState(prev => ({
                    ...prev,
                    isStreaming: false,
                    error: 'Connection failed'
                }));
            };

            wsRef.current = ws;
        } else {
            wsRef.current.send(JSON.stringify({
                query,
                include_sources: options?.includeSources ?? true,
                detailed: options?.detailed ?? true,
                conversation_id: options?.conversationId,
            }));
        }
    }, [handleEvent]);

    // Disconnect
    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
    }, []);

    // Reset state (for new conversations)
    const resetState = useCallback(() => {
        setStreamState({
            isStreaming: false,
            currentPhase: null,
            phases: [],
            chunksFound: null,
            response: '',
            sources: [],
            error: null,
            totalDuration: null,
        });
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    return {
        isConnected,
        streamState,
        sendQuery,
        connect,
        disconnect,
        resetState,
    };
}
