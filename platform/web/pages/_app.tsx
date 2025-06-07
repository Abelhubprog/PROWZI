
import type { AppProps } from 'next/app'
import { useEffect, useState } from 'react'
import Head from 'next/head'
import PWAInstaller from '../components/PWAInstaller'

interface ProwziTheme {
  colors: {
    primary: string
    secondary: string
    background: string
    surface: string
    text: string
  }
}

const defaultTheme: ProwziTheme = {
  colors: {
    primary: '#3B82F6',
    secondary: '#64748B', 
    background: '#0F172A',
    surface: '#1E293B',
    text: '#F8FAFC'
  }
}

export default function App({ Component, pageProps }: AppProps) {
  const [theme, setTheme] = useState<ProwziTheme>(defaultTheme)
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
    
    // Apply theme CSS variables
    const root = document.documentElement
    Object.entries(theme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value)
    })

    // Register service worker
    if ('serviceWorker' in navigator) {\n      window.addEventListener('load', () => {\n        navigator.serviceWorker.register('/sw.js')\n          .then((registration) => {\n            console.log('SW registered: ', registration);\n          })\n          .catch((registrationError) => {\n            console.log('SW registration failed: ', registrationError);\n          });\n      });\n    }
  }, [theme])

  if (!mounted) {
    return null
  }

  return (
    <>
      <Head>
        <title>Prowzi - AI Agent Intelligence Platform</title>
        <meta name="description" content="Autonomous AI agents for real-time intelligence gathering" />
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="theme-color" content="#3B82F6" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="Prowzi" />
        <meta name="mobile-web-app-capable" content="yes" />
        <link rel="manifest" href="/manifest.json" />
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/icons/icon-192x192.png" />
        <link rel="shortcut icon" href="/icons/icon-192x192.png" />
      </Head>
      
      <div className="prowzi-app">
        <Component {...pageProps} theme={theme} setTheme={setTheme} />
        <PWAInstaller />
      </div>

      <style jsx global>{`
        :root {
          --color-primary: ${theme.colors.primary};
          --color-secondary: ${theme.colors.secondary};
          --color-background: ${theme.colors.background};
          --color-surface: ${theme.colors.surface};
          --color-text: ${theme.colors.text};
        }

        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }

        body {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
          background: var(--color-background);
          color: var(--color-text);
          line-height: 1.6;
        }

        .prowzi-app {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
        }

        .loading-spinner {
          display: inline-block;
          width: 20px;
          height: 20px;
          border: 3px solid rgba(255, 255, 255, 0.3);
          border-radius: 50%;
          border-top-color: var(--color-primary);
          animation: spin 1s ease-in-out infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .error-boundary {
          padding: 2rem;
          text-align: center;
          background: var(--color-surface);
          border-radius: 8px;
          margin: 2rem;
        }

        .error-boundary h2 {
          color: #EF4444;
          margin-bottom: 1rem;
        }

        .btn {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 6px;
          background: var(--color-primary);
          color: white;
          cursor: pointer;
          font-weight: 500;
          transition: background-color 0.2s;
        }

        .btn:hover {
          background: #1E40AF;
        }

        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
      `}</style>
    </>
  )
}
