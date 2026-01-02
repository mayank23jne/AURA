import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Paper,
    Grid,
    CircularProgress
} from '@mui/material';
import {
    BarChart,
    Bar,
    LineChart,
    Line,
    PieChart,
    Pie,
    Cell,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend
} from 'recharts';
import { api } from '../api/client';

const Analytics: React.FC = () => {
    const [loading, setLoading] = useState(true);
    const [data, setData] = useState<any>(null);

    useEffect(() => {
        fetchAnalytics();
    }, []);

    const fetchAnalytics = async () => {
        setLoading(true);
        try {
            const [statusRes, auditsRes] = await Promise.all([
                api.getStatus(),
                api.getAudits()
            ]);
            setData({
                status: statusRes.data,
                audits: auditsRes.data.audits || []
            });
        } catch (error) {
            console.error('Error fetching analytics:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    // Prepare chart data
    const auditTrend = [
        { month: 'Jan', audits: 12 },
        { month: 'Feb', audits: 19 },
        { month: 'Mar', audits: 15 },
        { month: 'Apr', audits: 25 },
        { month: 'May', audits: 22 },
        { month: 'Jun', audits: data?.audits?.length || 18 }
    ];

    const complianceTrend = [
        { month: 'Jan', score: 75 },
        { month: 'Feb', score: 78 },
        { month: 'Mar', score: 82 },
        { month: 'Apr', score: 85 },
        { month: 'May', score: 88 },
        { month: 'Jun', score: 90 }
    ];

    const breachTypes = [
        { name: 'Security', value: 35, color: '#ef4444' },
        { name: 'Privacy', value: 25, color: '#f59e0b' },
        { name: 'Ethics', value: 20, color: '#8b5cf6' },
        { name: 'Bias', value: 20, color: '#3b82f6' }
    ];

    const agentPerformance = data?.status?.agents
        ? Object.entries(data.status.agents).map(([name, agent]: [string, any]) => ({
            name: name.charAt(0).toUpperCase() + name.slice(1),
            tasks: agent.metrics?.tasks_completed || 0
        }))
        : [];

    return (
        <Box sx={{ mt: 2 }}>
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" sx={{ fontWeight: 800, color: '#1e293b' }}>System Analytics</Typography>
                <Typography variant="body1" color="text.secondary">
                    Comprehensive insights into platform performance and compliance trends
                </Typography>
            </Box>

            <Grid container spacing={3}>
                {/* Audit Trend */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 3, borderRadius: 4 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>Audit Activity Trend</Typography>
                        <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={auditTrend}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="month" stroke="#64748b" />
                                <YAxis stroke="#64748b" />
                                <Tooltip />
                                <Line type="monotone" dataKey="audits" stroke="#6366f1" strokeWidth={3} />
                            </LineChart>
                        </ResponsiveContainer>
                    </Paper>
                </Grid>

                {/* Compliance Score Trend */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 3, borderRadius: 4 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>Compliance Score Trend</Typography>
                        <ResponsiveContainer width="100%" height={250}>
                            <LineChart data={complianceTrend}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="month" stroke="#64748b" />
                                <YAxis stroke="#64748b" domain={[0, 100]} />
                                <Tooltip />
                                <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={3} />
                            </LineChart>
                        </ResponsiveContainer>
                    </Paper>
                </Grid>

                {/* Breach Distribution */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 3, borderRadius: 4 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>Breach Type Distribution</Typography>
                        <ResponsiveContainer width="100%" height={250}>
                            <PieChart>
                                <Pie
                                    data={breachTypes}
                                    cx="50%"
                                    cy="50%"
                                    labelLine={false}
                                    label={(entry) => `${entry.name}: ${entry.value}%`}
                                    outerRadius={80}
                                    dataKey="value"
                                >
                                    {breachTypes.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip />
                            </PieChart>
                        </ResponsiveContainer>
                    </Paper>
                </Grid>

                {/* Agent Performance */}
                <Grid size={{ xs: 12, md: 6 }}>
                    <Paper sx={{ p: 3, borderRadius: 4 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>Agent Task Completion</Typography>
                        <ResponsiveContainer width="100%" height={250}>
                            <BarChart data={agentPerformance}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                                <XAxis dataKey="name" stroke="#64748b" />
                                <YAxis stroke="#64748b" />
                                <Tooltip />
                                <Bar dataKey="tasks" fill="#8b5cf6" radius={[8, 8, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </Paper>
                </Grid>

                {/* Summary Stats */}
                <Grid size={{ xs: 12 }}>
                    <Paper sx={{ p: 3, borderRadius: 4 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, mb: 3 }}>Platform Summary</Typography>
                        <Grid container spacing={3}>
                            <Grid item xs={12} sm={3}>
                                <Box sx={{ textAlign: 'center', p: 2, bgcolor: '#f8fafc', borderRadius: 2 }}>
                                    <Typography variant="h4" sx={{ fontWeight: 800, color: '#6366f1' }}>
                                        {data?.audits?.length || 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">Total Audits</Typography>
                                </Box>
                            </Grid>
                            <Grid item xs={12} sm={3}>
                                <Box sx={{ textAlign: 'center', p: 2, bgcolor: '#f8fafc', borderRadius: 2 }}>
                                    <Typography variant="h4" sx={{ fontWeight: 800, color: '#10b981' }}>
                                        {data?.status?.agents ? Object.keys(data.status.agents).length : 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">Active Agents</Typography>
                                </Box>
                            </Grid>
                            <Grid item xs={12} sm={3}>
                                <Box sx={{ textAlign: 'center', p: 2, bgcolor: '#f8fafc', borderRadius: 2 }}>
                                    <Typography variant="h4" sx={{ fontWeight: 800, color: '#f59e0b' }}>
                                        {data?.status?.knowledge_base?.total_items || 0}
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">Knowledge Items</Typography>
                                </Box>
                            </Grid>
                            <Grid size={{ xs: 12, sm: 3 }}>
                                <Box sx={{ textAlign: 'center', p: 2, bgcolor: '#f8fafc', borderRadius: 2 }}>
                                    <Typography variant="h4" sx={{ fontWeight: 800, color: '#8b5cf6' }}>
                                        90%
                                    </Typography>
                                    <Typography variant="body2" color="text.secondary">Avg Compliance</Typography>
                                </Box>
                            </Grid>
                        </Grid>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default Analytics;
