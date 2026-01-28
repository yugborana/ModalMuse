'use client';

// StreamProgress.tsx - Visual progress indicator for RAG pipeline phases

import { useState } from 'react';
import { PhaseEvent, ChunksFoundEvent, ChunkPreview } from '@/lib/useWebSocket';

interface StreamProgressProps {
    currentPhase: string | null;
    phases: PhaseEvent[];
    chunksFound: ChunksFoundEvent | null;
    isStreaming: boolean;
}

const PHASE_CONFIG = {
    embedding: { icon: '🔮', label: 'Embedding', color: '#a78bfa' },
    search: { icon: '🔍', label: 'Search', color: '#60a5fa' },
    reranking: { icon: '⚖️', label: 'Reranking', color: '#f59e0b' },
    fusion: { icon: '🔀', label: 'Fusion', color: '#10b981' },
    generation: { icon: '✨', label: 'Generating', color: '#b08d57' },
};

export default function StreamProgress({
    currentPhase,
    phases,
    chunksFound,
    isStreaming
}: StreamProgressProps) {
    const [expandedChunk, setExpandedChunk] = useState<string | null>(null);

    if (!isStreaming && phases.length === 0) return null;

    const completedPhases = new Set(phases.map(p => p.phase));

    return (
        <div className="glass-card p-4 mb-4 space-y-4 animate-fade-in">
            {/* Pipeline Progress */}
            <div className="flex items-center gap-2 text-xs">
                {Object.entries(PHASE_CONFIG).map(([phase, config], idx) => {
                    const isCompleted = completedPhases.has(phase as keyof typeof PHASE_CONFIG);
                    const isCurrent = currentPhase === phase;

                    return (
                        <div key={phase} className="flex items-center">
                            {idx > 0 && (
                                <div
                                    className="w-6 h-0.5 mx-1"
                                    style={{
                                        background: isCompleted
                                            ? config.color
                                            : 'var(--border-subtle)'
                                    }}
                                />
                            )}
                            <div
                                className={`
                                    flex items-center gap-1.5 px-2 py-1 rounded-full
                                    transition-all duration-300
                                    ${isCurrent ? 'ring-2 ring-offset-2 ring-offset-[var(--bg-primary)]' : ''}
                                `}
                                style={{
                                    background: isCompleted || isCurrent
                                        ? `${config.color}20`
                                        : 'transparent',
                                    color: isCompleted || isCurrent
                                        ? config.color
                                        : 'var(--text-muted)',
                                    borderColor: config.color,
                                    ...(isCurrent ? { '--tw-ring-color': config.color } as React.CSSProperties : {}),
                                }}
                            >
                                <span className={isCurrent ? 'animate-pulse' : ''}>
                                    {config.icon}
                                </span>
                                <span className="hidden sm:inline">{config.label}</span>
                                {isCompleted && (
                                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                        <path
                                            fillRule="evenodd"
                                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                                            clipRule="evenodd"
                                        />
                                    </svg>
                                )}
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* Chunks Found Preview */}
            {chunksFound && (
                <div className="border-t border-[var(--border-subtle)] pt-3">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-[var(--text-muted)]">
                            {chunksFound.message}
                        </span>
                    </div>

                    <div className="flex gap-2 overflow-x-auto pb-2">
                        {chunksFound.chunks.slice(0, 8).map((chunk, idx) => (
                            <ChunkCard
                                key={chunk.id || idx}
                                chunk={chunk}
                                isExpanded={expandedChunk === (chunk.id || String(idx))}
                                onToggle={() => setExpandedChunk(
                                    expandedChunk === (chunk.id || String(idx)) ? null : (chunk.id || String(idx))
                                )}
                            />
                        ))}
                    </div>
                </div>
            )}

            {/* Expanded Chunk View */}
            {expandedChunk && chunksFound && (
                <ExpandedChunkView
                    chunk={chunksFound.chunks.find(c => c.id === expandedChunk || String(chunksFound.chunks.indexOf(c)) === expandedChunk)}
                    onClose={() => setExpandedChunk(null)}
                />
            )}

            {/* Phase Timings */}
            {phases.length > 0 && !isStreaming && (
                <div className="flex flex-wrap gap-2 text-xs border-t border-[var(--border-subtle)] pt-3">
                    {phases.filter(p => p.duration_ms).map((phase, idx) => (
                        <span
                            key={idx}
                            className="px-2 py-1 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)]"
                        >
                            {PHASE_CONFIG[phase.phase]?.label}: {phase.duration_ms?.toFixed(0)}ms
                        </span>
                    ))}
                </div>
            )}
        </div>
    );
}

function ChunkCard({ chunk, isExpanded, onToggle }: {
    chunk: ChunkPreview;
    isExpanded: boolean;
    onToggle: () => void;
}) {
    const isImage = chunk.type === 'image';
    const imagePath = (chunk.metadata?.image_path || chunk.metadata?.url) as string | undefined;

    return (
        <button
            onClick={onToggle}
            className={`
                flex-shrink-0 p-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border-subtle)]
                hover:border-[var(--accent-gold)] hover:bg-[var(--bg-tertiary)] 
                transition-all duration-200 text-left
                ${isExpanded ? 'ring-2 ring-[var(--accent-gold)]' : ''}
                ${isImage ? 'w-24 h-24' : 'w-52'}
            `}
        >
            {isImage && imagePath ? (
                <div className="w-full h-full relative">
                    <img
                        src={imagePath.startsWith('http') ? imagePath : `/api/images/${encodeURIComponent(imagePath)}`}
                        alt="Source"
                        className="w-full h-full object-cover rounded"
                        onError={(e) => {
                            (e.target as HTMLImageElement).src = '/placeholder-image.png';
                        }}
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-black/60 px-1 py-0.5 rounded-b">
                        <span className="text-xs text-[var(--accent-gold)]">
                            {(chunk.score * 100).toFixed(0)}%
                        </span>
                    </div>
                </div>
            ) : (
                <>
                    <div className="flex items-center gap-1 mb-1">
                        <span className="text-xs">
                            {isImage ? '🖼️' : '📄'}
                        </span>
                        <span className="text-xs text-[var(--accent-gold)]">
                            {(chunk.score * 100).toFixed(0)}%
                        </span>
                    </div>
                    <p className="text-xs text-[var(--text-secondary)] line-clamp-3">
                        {chunk.preview}
                    </p>
                </>
            )}
        </button>
    );
}

function ExpandedChunkView({ chunk, onClose }: {
    chunk: ChunkPreview | undefined;
    onClose: () => void;
}) {
    if (!chunk) return null;

    const isImage = chunk.type === 'image';
    const imagePath = (chunk.metadata?.image_path || chunk.metadata?.url) as string | undefined;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
            <div className="glass-card max-w-2xl max-h-[80vh] overflow-auto p-6 relative">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 rounded-full bg-[var(--bg-secondary)] hover:bg-[var(--bg-tertiary)]"
                >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button>

                <div className="flex items-center gap-2 mb-4">
                    <span className="text-lg">{isImage ? '🖼️' : '📄'}</span>
                    <span className="text-sm font-medium text-[var(--accent-gold)]">
                        {(chunk.score * 100).toFixed(0)}% relevant
                    </span>
                    <span className="text-xs text-[var(--text-muted)]">
                        {isImage ? 'Image Source' : 'Text Source'}
                    </span>
                </div>

                {isImage && imagePath && (
                    <div className="mb-4">
                        <img
                            src={imagePath.startsWith('http') ? imagePath : `/api/images/${encodeURIComponent(imagePath)}`}
                            alt="Source"
                            className="max-w-full h-auto rounded-lg"
                            onError={(e) => {
                                (e.target as HTMLImageElement).style.display = 'none';
                            }}
                        />
                    </div>
                )}

                <div className="prose prose-invert prose-sm max-w-none">
                    <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap">
                        {chunk.preview}
                    </p>
                </div>

                {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
                    <div className="mt-4 pt-4 border-t border-[var(--border-subtle)]">
                        <p className="text-xs text-[var(--text-muted)] mb-2">Metadata:</p>
                        <div className="text-xs text-[var(--text-secondary)] space-y-1">
                            {Object.entries(chunk.metadata).slice(0, 5).map(([key, value]) => (
                                <div key={key} className="flex gap-2">
                                    <span className="text-[var(--text-muted)]">{key}:</span>
                                    <span className="truncate">{String(value ?? '').slice(0, 100)}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
