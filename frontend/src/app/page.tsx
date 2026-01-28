'use client';

import { useState, useEffect, useCallback } from 'react';
import ChatInterface from '@/components/ChatInterface';
import UploadModal from '@/components/UploadModal';
import ConversationSidebar from '@/components/ConversationSidebar';
import { getCollectionStats, CollectionStats } from '@/lib/api';
import { getConversation, ConversationWithMessages } from '@/lib/conversations';

export default function Home() {
  const [showUpload, setShowUpload] = useState(false);
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false); // Start closed for cleaner look
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [currentConversation, setCurrentConversation] = useState<ConversationWithMessages | null>(null);

  useEffect(() => {
    setIsLoaded(true);

    const fetchStats = async () => {
      try {
        const data = await getCollectionStats();
        setStats(data);
      } catch (error) {
        console.error('Failed to fetch stats:', error);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  // Load conversation when ID changes
  useEffect(() => {
    if (currentConversationId) {
      loadConversation(currentConversationId);
    } else {
      setCurrentConversation(null);
    }
  }, [currentConversationId]);

  const loadConversation = async (id: string) => {
    try {
      const data = await getConversation(id);
      setCurrentConversation(data);
    } catch (error) {
      console.error('Failed to load conversation:', error);
      setCurrentConversationId(null);
    }
  };

  const handleSelectConversation = useCallback((id: string) => {
    setCurrentConversationId(id);
    setSidebarOpen(false);
  }, []);

  const handleNewConversation = useCallback(() => {
    setCurrentConversationId(null);
    setCurrentConversation(null);
  }, []);

  return (
    <main className="min-h-screen">
      {/* Conversation Sidebar - Fixed overlay, doesn't affect layout */}
      <ConversationSidebar
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
      />

      {/* Header - Always full width */}
      <header className={`fixed top-0 left-0 right-0 z-10 bg-[var(--bg-primary)]/80 backdrop-blur-lg border-b border-[var(--border-subtle)] transition-all duration-300 ${isLoaded ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'}`}>
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          {/* Logo - Click to go home */}
          <button
            onClick={handleNewConversation}
            className="flex items-center gap-3 hover:opacity-80 transition-opacity"
          >
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-[var(--accent-gold)] to-[var(--accent-gold-light)] flex items-center justify-center">
              <svg className="w-5 h-5 text-[var(--bg-primary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
              </svg>
            </div>
            <span className="text-lg font-medium tracking-tight">ModalMuse</span>
          </button>

          {/* Stats */}
          <div className="hidden md:flex items-center gap-6 text-sm">
            {stats ? (
              <>
                <div className="flex items-center gap-2 text-[var(--text-secondary)]">
                  <svg className="w-4 h-4 text-[var(--accent-cyan)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span>{stats.text_collection.points_count || 0} chunks</span>
                </div>
                <div className="flex items-center gap-2 text-[var(--text-secondary)]">
                  <svg className="w-4 h-4 text-[var(--accent-gold)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span>{stats.image_collection.points_count || 0} images</span>
                </div>
              </>
            ) : (
              <span className="text-[var(--text-muted)]">Loading...</span>
            )}
          </div>

          {/* Upload CTA */}
          <button
            onClick={() => setShowUpload(true)}
            className="btn-outline flex items-center gap-2 text-xs py-2 px-4"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span className="hidden sm:inline">Upload</span>
          </button>
        </div>
      </header>

      {/* Main Chat Interface - Full width, centered */}
      <ChatInterface
        conversationId={currentConversationId}
        initialMessages={currentConversation?.messages}
        onConversationCreated={setCurrentConversationId}
      />

      {/* Upload Modal */}
      {showUpload && (
        <UploadModal onClose={() => setShowUpload(false)} />
      )}
    </main>
  );
}
