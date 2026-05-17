// API client for ModalMuse backend
// Only includes endpoints actively used by the frontend.
// Query operations use WebSocket (see useWebSocket.ts).

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
    return { task_id: result.task_id || '' };
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
