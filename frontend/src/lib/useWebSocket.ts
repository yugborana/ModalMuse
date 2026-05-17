// WebSocket hook for real-time RAG streaming with auto-reconnection
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
    phase: 'embedding' | 'search' | 'fusion' | 'reranking' | 'generation';
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
// RECONNECTION CONFIG
// ═══════════════════════════════════════════════════════════════════
const MAX_RECONNECT_ATTEMPTS = 5;
const BASE_RECONNECT_DELAY_MS = 1000; // 1s, 2s, 4s, 8s, 16s

// ═══════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════

const INITIAL_STATE: StreamState = {
    isStreaming: false,
    currentPhase: null,
    phases: [],
    chunksFound: null,
    response: '',
    sources: [],
    error: null,
    totalDuration: null,
};

export function useRAGWebSocket() {
    const wsRef = useRef<WebSocket | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const reconnectAttemptsRef = useRef(0);
    const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const pendingQueryRef = useRef<{ query: string; options?: { includeSources?: boolean; conversationId?: string } } | null>(null);

    const [streamState, setStreamState] = useState<StreamState>({ ...INITIAL_STATE });

    // ── Event handler (stable, no deps) ──
    const handleEventRef = useRef<(event: StreamEvent) => void>(() => {});
    handleEventRef.current = (event: StreamEvent) => {
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
    };

    // ── Connection factory (ref-based to avoid circular useCallback) ──
    const createConnectionRef = useRef<() => WebSocket>(() => null as unknown as WebSocket);
    createConnectionRef.current = (): WebSocket => {
        const ws = new WebSocket(`${WS_BASE}/ws/query`);

        ws.onopen = () => {
            console.log('[WS] Connected');
            setIsConnected(true);
            reconnectAttemptsRef.current = 0;

            if (pendingQueryRef.current) {
                const { query, options } = pendingQueryRef.current;
                pendingQueryRef.current = null;
                ws.send(JSON.stringify({
                    query,
                    include_sources: options?.includeSources ?? true,
                    detailed: true,
                    conversation_id: options?.conversationId,
                }));
            }
        };

        ws.onmessage = (msgEvent) => {
            try {
                const data: StreamEvent = JSON.parse(msgEvent.data);
                handleEventRef.current(data);
            } catch (e) {
                console.error('[WS] Parse error:', e);
            }
        };

        ws.onclose = (closeEvent) => {
            console.log('[WS] Connection closed:', closeEvent.code, closeEvent.reason);
            setIsConnected(false);
            wsRef.current = null;

            // Auto-reconnect only on unexpected close (not code 1000)
            if (closeEvent.code !== 1000 && reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
                const delay = BASE_RECONNECT_DELAY_MS * Math.pow(2, reconnectAttemptsRef.current);
                console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current + 1}/${MAX_RECONNECT_ATTEMPTS})`);
                reconnectTimerRef.current = setTimeout(() => {
                    reconnectAttemptsRef.current += 1;
                    wsRef.current = createConnectionRef.current();
                }, delay);
            } else if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
                setStreamState(prev => ({
                    ...prev,
                    isStreaming: false,
                    error: 'Connection lost. Please refresh the page.',
                }));
            }
        };

        ws.onerror = () => {
            console.error('[WS] Connection error');
        };

        return ws;
    };

    // ── Public API (all stable references) ──

    const sendQuery = useCallback((query: string, options?: {
        includeSources?: boolean;
        conversationId?: string;
    }) => {
        setStreamState({
            ...INITIAL_STATE,
            isStreaming: true,
        });

        const message = JSON.stringify({
            query,
            include_sources: options?.includeSources ?? true,
            detailed: true,
            conversation_id: options?.conversationId,
        });

        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(message);
        } else {
            pendingQueryRef.current = { query, options };
            wsRef.current = createConnectionRef.current();
        }
    }, []);

    const disconnect = useCallback(() => {
        if (reconnectTimerRef.current) {
            clearTimeout(reconnectTimerRef.current);
            reconnectTimerRef.current = null;
        }
        reconnectAttemptsRef.current = MAX_RECONNECT_ATTEMPTS;
        if (wsRef.current) {
            wsRef.current.close(1000, 'Client disconnect');
            wsRef.current = null;
        }
    }, []);

    const resetState = useCallback(() => {
        setStreamState({ ...INITIAL_STATE });
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
        disconnect,
        resetState,
    };
}
