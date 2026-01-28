'use client';

import { useState, useCallback } from 'react';
import { uploadDocument, getIndexingStatus, IndexingStatus } from '@/lib/api';

interface UploadModalProps {
    onClose: () => void;
}

export default function UploadModal({ onClose }: UploadModalProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<IndexingStatus | null>(null);
    const [isPolling, setIsPolling] = useState(false);

    const pollStatus = useCallback(async (taskId: string) => {
        setIsPolling(true);

        const poll = async () => {
            try {
                const status = await getIndexingStatus(taskId);
                setUploadStatus(status);

                if (status.status === 'completed' || status.status === 'failed') {
                    setIsPolling(false);
                    return;
                }

                setTimeout(poll, 2000);
            } catch (error) {
                console.error('Polling error:', error);
                setIsPolling(false);
            }
        };

        poll();
    }, []);

    const handleUpload = async (file: File) => {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            alert('Only PDF files are supported');
            return;
        }

        setUploadStatus({
            status: 'started',
            message: 'Uploading document...',
            file_name: file.name,
            progress: 5,
        });

        try {
            const { task_id } = await uploadDocument(file);
            setUploadStatus({
                status: 'processing',
                message: 'Processing with AI...',
                file_name: file.name,
                progress: 15,
            });

            if (task_id) {
                pollStatus(task_id);
            }
        } catch (error) {
            setUploadStatus({
                status: 'failed',
                message: 'Upload failed',
                error: error instanceof Error ? error.message : 'Unknown error',
                file_name: file.name,
            });
        }
    };

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        if (file) handleUpload(file);
    }, []);

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) handleUpload(file);
    };

    return (
        <>
            {/* Backdrop */}
            <div
                className="fixed inset-0 bg-black/70 backdrop-blur-sm z-40 animate-fade-in"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[520px] max-w-[95vw] z-50 animate-fade-in-up">
                <div className="bg-[var(--bg-secondary)] border border-[var(--border-subtle)] rounded-2xl overflow-hidden shadow-2xl">
                    {/* Header */}
                    <div className="flex items-center justify-between p-6 border-b border-[var(--border-subtle)]">
                        <div>
                            <h2 className="font-serif text-xl font-medium">Upload Document</h2>
                            <p className="text-sm text-[var(--text-muted)] mt-1">
                                Add PDFs to your knowledge base
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
                    <div className="p-6">
                        {/* Drop Zone */}
                        <div
                            onDrop={handleDrop}
                            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                            onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
                            className={`
                relative border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer
                ${isDragging
                                    ? 'border-[var(--accent-gold)] bg-[var(--accent-gold)]/5'
                                    : 'border-[var(--border-medium)] hover:border-[var(--accent-gold)]/50 hover:bg-[var(--glass-hover)]'
                                }
              `}
                        >
                            <input
                                type="file"
                                accept=".pdf"
                                onChange={handleFileSelect}
                                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                                disabled={isPolling}
                            />

                            <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center transition-colors ${isDragging ? 'bg-[var(--accent-gold)]/20' : 'bg-[var(--glass-bg)]'
                                }`}>
                                <svg
                                    className={`w-8 h-8 transition-colors ${isDragging ? 'text-[var(--accent-gold)]' : 'text-[var(--text-muted)]'}`}
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                >
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                            </div>

                            <p className="text-[var(--text-primary)] font-medium mb-1">
                                {isPolling ? 'Processing...' : 'Drop your PDF here'}
                            </p>
                            <p className="text-sm text-[var(--text-muted)]">
                                or click to browse • Max 50MB
                            </p>
                        </div>

                        {/* Status */}
                        {uploadStatus && (
                            <div className="mt-6 p-5 glass-card">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-2 h-2 rounded-full ${uploadStatus.status === 'completed' ? 'bg-green-400' :
                                                uploadStatus.status === 'failed' ? 'bg-red-400' :
                                                    'bg-[var(--accent-gold)] animate-pulse'
                                            }`} />
                                        <span className="text-sm font-medium truncate max-w-[200px]">
                                            {uploadStatus.file_name}
                                        </span>
                                    </div>
                                    <span className={`text-xs uppercase tracking-wider ${uploadStatus.status === 'completed' ? 'text-green-400' :
                                            uploadStatus.status === 'failed' ? 'text-red-400' :
                                                'text-[var(--accent-gold)]'
                                        }`}>
                                        {uploadStatus.status}
                                    </span>
                                </div>

                                {/* Progress Bar */}
                                {(uploadStatus.status === 'started' || uploadStatus.status === 'processing') && (
                                    <div className="h-1 bg-[var(--bg-elevated)] rounded-full overflow-hidden mb-3">
                                        <div
                                            className="h-full bg-gradient-to-r from-[var(--accent-gold)] to-[var(--accent-gold-light)] transition-all duration-700"
                                            style={{ width: `${uploadStatus.progress || 0}%` }}
                                        />
                                    </div>
                                )}

                                <p className="text-sm text-[var(--text-secondary)]">
                                    {uploadStatus.message}
                                </p>

                                {uploadStatus.status === 'completed' && (
                                    <div className="mt-4 flex gap-6 text-sm">
                                        <div className="flex items-center gap-2">
                                            <svg className="w-4 h-4 text-[var(--accent-cyan)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                            </svg>
                                            <span className="text-[var(--text-secondary)]">{uploadStatus.text_vectors || 0} text chunks</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            <svg className="w-4 h-4 text-[var(--accent-gold)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                            </svg>
                                            <span className="text-[var(--text-secondary)]">{uploadStatus.image_vectors || 0} images</span>
                                        </div>
                                    </div>
                                )}

                                {uploadStatus.error && (
                                    <p className="mt-3 text-sm text-red-400">{uploadStatus.error}</p>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}
