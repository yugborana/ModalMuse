'use client';

import { useState } from 'react';
import { SourceNode } from '@/lib/api';

interface SourcesPanelProps {
    sources: SourceNode[] | null;
    isOpen: boolean;
    onClose: () => void;
}

export default function SourcesPanel({ sources, isOpen, onClose }: SourcesPanelProps) {
    const [selectedImage, setSelectedImage] = useState<string | null>(null);

    if (!isOpen) return null;

    const textSources = sources?.filter((s) => s.type === 'text') || [];
    const imageSources = sources?.filter((s) => s.type === 'image') || [];

    return (
        <>
            {/* Backdrop */}
            <div
                className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 animate-fade-in"
                onClick={onClose}
            />

            {/* Panel */}
            <div className="fixed right-0 top-0 h-full w-[480px] max-w-[95vw] bg-[var(--bg-secondary)] border-l border-[var(--border-subtle)] z-50 animate-slide-in-right overflow-hidden flex flex-col">
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-[var(--border-subtle)]">
                    <div>
                        <h2 className="font-serif text-xl font-medium">Sources</h2>
                        <p className="text-sm text-[var(--text-muted)] mt-1">
                            {sources?.length || 0} relevant items found
                        </p>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 hover:bg-[var(--glass-hover)] rounded-lg transition-colors"
                    >
                        <svg className="w-5 h-5 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-y-auto p-6 space-y-8">
                    {/* Image Sources */}
                    {imageSources.length > 0 && (
                        <div>
                            <h3 className="text-xs font-medium text-[var(--accent-gold)] uppercase tracking-wider mb-4 flex items-center gap-2">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                                Images ({imageSources.length})
                            </h3>
                            <div className="grid grid-cols-2 gap-4">
                                {imageSources.map((source, index) => {
                                    // Priority: metadata.image_url (Supabase) > content > metadata.image_path
                                    const metadata = source.metadata as Record<string, string>;
                                    const imagePath = metadata?.image_url ||
                                        source.content ||
                                        metadata?.image_path || '';
                                    // Check if it's already a URL (Supabase or LlamaParse) or a local path
                                    const isUrl = imagePath.startsWith('http://') || imagePath.startsWith('https://');
                                    const imageUrl = isUrl
                                        ? imagePath
                                        : `http://localhost:8000/downloaded_images/${imagePath.split(/[\\/]/).pop() || ''}`;

                                    return (
                                        <div
                                            key={index}
                                            className="glass-card glass-card-hover cursor-pointer overflow-hidden group"
                                            onClick={() => setSelectedImage(imageUrl)}
                                        >
                                            <div className="aspect-square relative bg-[var(--bg-elevated)]">
                                                <img
                                                    src={imageUrl}
                                                    alt={`Source ${index + 1}`}
                                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                                                    onError={(e) => {
                                                        (e.target as HTMLImageElement).style.display = 'none';
                                                    }}
                                                />
                                                <div className="absolute inset-0 bg-gradient-to-t from-black/70 to-transparent opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-3">
                                                    <span className="text-xs text-white font-medium">
                                                        {(source.score * 100).toFixed(0)}% match
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}

                    {/* Text Sources */}
                    {textSources.length > 0 && (
                        <div>
                            <h3 className="text-xs font-medium text-[var(--accent-cyan)] uppercase tracking-wider mb-4 flex items-center gap-2">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                                Text Chunks ({textSources.length})
                            </h3>
                            <div className="space-y-4">
                                {textSources.map((source, index) => (
                                    <div key={index} className="glass-card p-5">
                                        <div className="flex items-center justify-between mb-3">
                                            <div className="flex items-center gap-2">
                                                <div className="w-6 h-6 rounded-full bg-[var(--accent-cyan)]/10 flex items-center justify-center text-xs text-[var(--accent-cyan)] font-medium">
                                                    {index + 1}
                                                </div>
                                                <span className="text-xs font-medium text-[var(--accent-cyan)]">
                                                    {(source.score * 100).toFixed(0)}% relevant
                                                </span>
                                            </div>
                                            {typeof source.metadata?.source === 'string' && source.metadata.source && (
                                                <span className="text-xs text-[var(--text-muted)] truncate max-w-[120px]">
                                                    {source.metadata.source.split(/[\\/]/).pop()}
                                                </span>
                                            )}
                                        </div>
                                        <p className="text-sm text-[var(--text-secondary)] leading-relaxed">
                                            {source.content.length > 300
                                                ? source.content.slice(0, 300) + '...'
                                                : source.content
                                            }
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {(!sources || sources.length === 0) && (
                        <div className="flex flex-col items-center justify-center py-16 text-center">
                            <div className="w-16 h-16 rounded-full bg-[var(--glass-bg)] flex items-center justify-center mb-4">
                                <svg className="w-8 h-8 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                            </div>
                            <p className="text-[var(--text-muted)]">No sources available</p>
                        </div>
                    )}
                </div>
            </div>

            {/* Image Lightbox */}
            {selectedImage && (
                <div
                    className="fixed inset-0 bg-black/95 z-[60] flex items-center justify-center p-8 animate-fade-in"
                    onClick={() => setSelectedImage(null)}
                >
                    <button
                        className="absolute top-6 right-6 p-3 text-white hover:bg-white/10 rounded-full transition-colors"
                        onClick={() => setSelectedImage(null)}
                    >
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                    <img
                        src={selectedImage}
                        alt="Source image"
                        className="max-h-[85vh] max-w-[90vw] object-contain rounded-lg shadow-2xl"
                        onClick={(e) => e.stopPropagation()}
                    />
                </div>
            )}
        </>
    );
}
