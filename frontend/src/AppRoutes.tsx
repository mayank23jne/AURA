import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Agent from './pages/Agent';
import Model from './pages/Model';
import Audits from './pages/Audits';
import Policy from './pages/Policy';
import Packages from './pages/Packages';
import Reports from './pages/Reports';
import Analytics from './pages/Analytics';
import GenericModule from './pages/GenericModule';

const AppRoutes: React.FC = () => {
    return (
        <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/ai-dashboard" element={<Dashboard />} />
            <Route path="/agent" element={<Agent />} />
            <Route path="/model" element={<Model />} />
            <Route path="/audits" element={<Audits />} />
            <Route path="/policy" element={<Policy />} />
            <Route path="/packages" element={<Packages />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/comparison" element={<GenericModule title="Model Comparison" />} />
            <Route path="/metrics" element={<GenericModule title="Performance Metrics" />} />
            <Route path="/settings" element={<GenericModule title="System Settings" />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
    );
};

export default AppRoutes;
