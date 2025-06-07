import React from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import ProtectionDashboard from '../components/ProtectionDashboard';

export default function ProtectionPage() {
  return (
    <>
      <Head>
        <title>Protection Dashboard - Prowzi</title>
        <meta name="description" content="Advanced trading protection and risk management" />
      </Head>

      <div className="min-h-screen bg-slate-900">
        {/* Header */}
        <header className="bg-slate-800 border-b border-slate-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-4">
                <Link href="/" className="text-slate-400 hover:text-white">
                  <ArrowLeft className="h-6 w-6" />
                </Link>
                <h1 className="text-xl font-semibold text-white">Protection Dashboard</h1>
              </div>
            </div>
          </div>
        </header>

        {/* Protection Dashboard Component */}
        <ProtectionDashboard />
      </div>
    </>
  );
}