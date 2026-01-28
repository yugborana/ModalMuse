// API client for ModalMuse backend

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

export interface SourceNode {
    content: string;
    score: number;
    type: 'text' | 'image';
    metadata: Record<string, unknown>;
}

export interface QueryResponse {
    answer: string;
    sources: SourceNode[] | null;
    query: string;
}

export interface IndexingStatus {
    status: 'started' | 'processing' | 'completed' | 'failed';
    message: string;
    file_name?: string;
    text_vectors?: number;
    image_vectors?: number;
    error?: string;
    progress?: number;
    from_cache?: boolean;
}

export interface CollectionStats {
    text_collection: {
        name: string;
        points_count: number;
        status: string;
    };
    image_collection: {
        name: string;
        points_count: number;
        status: string;
    };
}

// ═══════════════════════════════════════════════════════════════════
// QUERY API
// ═══════════════════════════════════════════════════════════════════

export async function queryRAG(query: string): Promise<QueryResponse> {
    const response = await fetch(`${API_BASE}/api/query/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, include_sources: true }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Query failed');
    }

    return response.json();
}

// Streaming query (returns async generator)
export async function* queryRAGStream(query: string): AsyncGenerator<string, void, unknown> {
    const response = await fetch(`${API_BASE}/api/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });

    if (!response.ok) {
        throw new Error('Stream query failed');
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                try {
                    const parsed = JSON.parse(data);
                    if (parsed.chunk) yield parsed.chunk;
                } catch {
                    yield data;
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// INDEXING API
// ═══════════════════════════════════════════════════════════════════

export async function uploadDocument(file: File): Promise<{ task_id: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/api/index/upload`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
    }

    const result = await response.json();
    // Extract task_id from message
    const match = result.message?.match(/Task ID: ([a-f0-9-]+)/);
    return { task_id: match?.[1] || '' };
}

export async function getIndexingStatus(taskId: string): Promise<IndexingStatus> {
    const response = await fetch(`${API_BASE}/api/index/status/${taskId}`);

    if (!response.ok) {
        throw new Error('Failed to get status');
    }

    return response.json();
}

export async function getCollectionStats(): Promise<CollectionStats> {
    try {
        const response = await fetch(`${API_BASE}/api/index/collections`);

        if (!response.ok) {
            throw new Error('Failed to get stats');
        }

        return response.json();
    } catch {
        // Return default empty stats on error (backend may not be running)
        return {
            text_collection: { name: 'mm_text', points_count: 0, status: 'unknown' },
            image_collection: { name: 'mm_images', points_count: 0, status: 'unknown' },
        };
    }
}

// ═══════════════════════════════════════════════════════════════════
// HEALTH API
// ═══════════════════════════════════════════════════════════════════

export async function getHealth(): Promise<Record<string, unknown>> {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
}
