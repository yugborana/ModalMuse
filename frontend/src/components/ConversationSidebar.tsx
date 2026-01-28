'use client';

// ConversationSidebar.tsx - Sidebar showing conversation history

import { useState, useEffect } from 'react';
import { Conversation, getConversations, createConversation, deleteConversation } from '@/lib/conversations';

interface ConversationSidebarProps {
    currentConversationId: string | null;
    onSelectConversation: (id: string) => void;
    onNewConversation: () => void;
    isOpen: boolean;
    onToggle: () => void;
}

export default function ConversationSidebar({
    currentConversationId,
    onSelectConversation,
    onNewConversation,
    isOpen,
    onToggle,
}: ConversationSidebarProps) {
    const [conversations, setConversations] = useState<Conversation[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Fetch conversations on mount
    useEffect(() => {
        fetchConversations();
    }, []);

    const fetchConversations = async () => {
        try {
            setIsLoading(true);
            const data = await getConversations();
            setConversations(data);
            setError(null);
        } catch (err) {
            console.error('Failed to fetch conversations:', err);
            setError('Failed to load conversations');
        } finally {
            setIsLoading(false);
        }
    };

    const handleNewConversation = async () => {
        try {
            const newConv = await createConversation();
            setConversations(prev => [newConv, ...prev]);
            onSelectConversation(newConv.id);
        } catch (err) {
            console.error('Failed to create conversation:', err);
        }
    };

    const handleDeleteConversation = async (id: string, e: React.MouseEvent) => {
        e.stopPropagation();
        if (!confirm('Delete this conversation?')) return;

        try {
            await deleteConversation(id);
            setConversations(prev => prev.filter(c => c.id !== id));
            if (currentConversationId === id) {
                onNewConversation();
            }
        } catch (err) {
            console.error('Failed to delete conversation:', err);
        }
    };

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        const now = new Date();
        const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24));

        if (diffDays === 0) return 'Today';
        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 7) return `${diffDays} days ago`;
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    return (
        <>
            {/* Toggle Arrow Button - ALWAYS VISIBLE */}
            <button
                onClick={onToggle}
                className={`
                    fixed top-1/2 -translate-y-1/2 z-30
                    w-6 h-20 rounded-r-lg
                    bg-[var(--bg-secondary)] border border-l-0 border-[var(--border-subtle)]
                    flex items-center justify-center
                    hover:bg-[var(--glass-hover)] transition-all duration-300
                    shadow-lg
                `}
                style={{ left: isOpen ? '288px' : '0px' }}
                title={isOpen ? 'Close history' : 'Open history'}
            >
                <svg
                    className={`w-4 h-4 text-[var(--text-muted)] transition-transform duration-300 ${isOpen ? '' : 'rotate-180'}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
            </button>

            {/* Sidebar Panel */}
            <div
                className={`
                    fixed inset-y-0 left-0 z-20
                    w-72 bg-[var(--bg-secondary)] border-r border-[var(--border-subtle)]
                    transform transition-transform duration-300 ease-in-out
                    ${isOpen ? 'translate-x-0' : '-translate-x-full'}
                    pt-20 flex flex-col
                `}
            >
                {/* Header */}
                <div className="p-4 border-b border-[var(--border-subtle)]">
                    <button
                        onClick={handleNewConversation}
                        className="w-full btn-primary text-sm flex items-center justify-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        New Chat
                    </button>
                </div>

                {/* Conversations List */}
                <div className="flex-1 overflow-y-auto p-2">
                    {isLoading ? (
                        <div className="flex justify-center py-8">
                            <div className="spinner" />
                        </div>
                    ) : error ? (
                        <div className="text-center py-8 text-[var(--text-muted)] text-sm">
                            {error}
                            <button
                                onClick={fetchConversations}
                                className="block mx-auto mt-2 text-[var(--accent-gold)] hover:underline"
                            >
                                Retry
                            </button>
                        </div>
                    ) : conversations.length === 0 ? (
                        <div className="text-center py-8 text-[var(--text-muted)] text-sm">
                            No conversations yet
                        </div>
                    ) : (
                        <div className="space-y-1">
                            {conversations.map((conv) => (
                                <div
                                    key={conv.id}
                                    onClick={() => onSelectConversation(conv.id)}
                                    className={`
                                        group flex items-center justify-between
                                        px-3 py-2.5 rounded-lg cursor-pointer
                                        transition-colors duration-200
                                        ${currentConversationId === conv.id
                                            ? 'bg-[var(--accent-gold)]/10 border border-[var(--accent-gold)]/30'
                                            : 'hover:bg-[var(--glass-hover)]'
                                        }
                                    `}
                                >
                                    <div className="flex-1 min-w-0">
                                        <p className={`
                                            text-sm truncate
                                            ${currentConversationId === conv.id
                                                ? 'text-[var(--accent-gold)]'
                                                : 'text-[var(--text-secondary)]'
                                            }
                                        `}>
                                            {conv.title || 'New Chat'}
                                        </p>
                                        <p className="text-xs text-[var(--text-muted)] mt-0.5">
                                            {formatDate(conv.updated_at)}
                                        </p>
                                    </div>

                                    {/* Delete Button */}
                                    <button
                                        onClick={(e) => handleDeleteConversation(conv.id, e)}
                                        className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 rounded transition-all"
                                        title="Delete conversation"
                                    >
                                        <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                        </svg>
                                    </button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className="p-4 border-t border-[var(--border-subtle)]">
                    <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        Supabase Connected
                    </div>
                </div>
            </div>

            {/* Backdrop when open on mobile */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-10 lg:hidden"
                    onClick={onToggle}
                />
            )}
        </>
    );
}
