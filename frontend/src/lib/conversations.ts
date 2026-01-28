// Frontend API client for conversations
// lib/conversations.ts

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ═══════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════

export interface Conversation {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
}

export interface Message {
    id: string;
    conversation_id: string;
    role: 'user' | 'assistant';
    content: string;
    sources?: SourceItem[] | null;
    created_at: string;
}

export interface SourceItem {
    content: string;
    score: number;
    type: 'text' | 'image';
    metadata: Record<string, unknown>;
}

export interface ConversationWithMessages extends Conversation {
    messages: Message[];
}

// ═══════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/**
 * Fetch all conversations (most recent first)
 */
export async function getConversations(limit = 50): Promise<Conversation[]> {
    const response = await fetch(`${API_BASE}/api/conversations?limit=${limit}`);

    if (!response.ok) {
        throw new Error('Failed to fetch conversations');
    }

    return response.json();
}

/**
 * Create a new conversation
 */
export async function createConversation(title?: string): Promise<Conversation> {
    const response = await fetch(`${API_BASE}/api/conversations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
    });

    if (!response.ok) {
        throw new Error('Failed to create conversation');
    }

    return response.json();
}

/**
 * Get a conversation with its messages
 */
export async function getConversation(id: string): Promise<ConversationWithMessages> {
    const response = await fetch(`${API_BASE}/api/conversations/${id}`);

    if (!response.ok) {
        if (response.status === 404) {
            throw new Error('Conversation not found');
        }
        throw new Error('Failed to fetch conversation');
    }

    return response.json();
}

/**
 * Add a message to a conversation
 */
export async function addMessage(
    conversationId: string,
    role: 'user' | 'assistant',
    content: string,
    sources?: SourceItem[]
): Promise<Message> {
    const response = await fetch(`${API_BASE}/api/conversations/${conversationId}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role, content, sources }),
    });

    if (!response.ok) {
        throw new Error('Failed to add message');
    }

    return response.json();
}

/**
 * Delete a conversation
 */
export async function deleteConversation(id: string): Promise<void> {
    const response = await fetch(`${API_BASE}/api/conversations/${id}`, {
        method: 'DELETE',
    });

    if (!response.ok) {
        throw new Error('Failed to delete conversation');
    }
}

/**
 * Update conversation title (auto-generate from first message)
 */
export function generateTitle(firstMessage: string): string {
    // Take first 50 chars or first sentence
    const truncated = firstMessage.slice(0, 50);
    const firstSentence = truncated.split(/[.!?]/)[0];
    return firstSentence.length > 0 ? firstSentence : truncated;
}
