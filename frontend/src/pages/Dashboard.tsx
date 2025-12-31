import React, { useState, useEffect } from 'react';
import {
    Box,
    Grid,
    Paper,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    CircularProgress
} from '@mui/material';
import {
    Timer as TimerIcon
} from '@mui/icons-material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Cell } from 'recharts';
import { api } from '../api/client';
import type { Agent, AgentsMap, DashboardData } from '../Types'; 

const StatsRow = ({ data }: { data: DashboardData | null  }) => {
    const stats = [
        {
            label: 'Total Agents',
            value: data?.agents ? Object.keys(data.agents).length : 12,
            trend: '+2 vs last Month',
            icon: <img src='/Icon.png'/>,
            iconBg: '#f5f3ff'
        },
        {
            label: 'Active Agents',
            value: data?.agents ? Object.values(data.agents).filter((a: Agent) => a.status === 'active' || a.status === 'idle').length : 10,
            trend: '+2 vs last Month',
            icon: <img src='/Icon (1).png'/>,
            iconBg: '#eff6ff'
        },
        {
            label: 'Knowledge Items',
            value: data?.knowledge_base?.total_items || 12,
            trend: '+8 vs last Month',
            icon: <img src='/Icon (3).png'/>,
            iconBg: '#eef2ff'
        },
        {
            label: 'Total Events',
            value: data?.event_stream?.total_events || 12,
            trend: '-5 vs last Month',
            icon: <img src='/Icon (4).png'/>,
            iconBg: '#fff1f2'
        },
    ];

    return (
        <Grid container spacing={2} sx={{ mb: 4 }}>
            {stats.map((stat) => (
                <Grid key={stat.label} size={{ xs: 12, md: 3 }}>
                    <Paper sx={{ p: 2.5, display: 'flex', alignItems: 'center', gap: 2, boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                        <Box sx={{
                            borderRadius: '12px',
                            bgcolor: stat.iconBg,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            {stat.icon}
                        </Box>
                        <Box>
                            <Typography variant="body2" sx={{ fontWeight: 500, fontSize: '0.85rem', color:'#898989' }}>{stat.label}</Typography>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="h5" sx={{ fontWeight: 800, my: 0.2 }}>{stat.value}</Typography>
                                <Typography variant="caption" sx={{ color: stat.trend.includes('-') ? '#ef4444' : '#10b981', fontWeight: 700 }}>
                                    {stat.trend.split(' ')[0]}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500 }}>
                                    {stat.trend.split(' ').slice(1).join(' ')}
                                </Typography>
                            </Box>
                        </Box>
                    </Paper>
                </Grid>
            ))}
        </Grid>
    );
};


const AgentsTable = ({ agents }: { agents?: AgentsMap  }) => {
    type AgentRow = Agent & { name: string };
    const agentList: AgentRow[] =
  agents && Object.keys(agents).length > 0
    ? Object.entries(agents).map(([name, data]) => ({
        name,
        ...data,
    })) : [
        { name: 'Orchestrator', status: 'active', metrics: { tasks_completed: 142, tasks_failed: 3, avg_response_time: 245 } },
        { name: 'Policy', status: 'active', metrics: { tasks_completed: 98, tasks_failed: 1, avg_response_time: 198 } },
        { name: 'Audit', status: 'inactive', metrics: { tasks_completed: 67, tasks_failed: 8, avg_response_time: 312 } },
        { name: 'Testing', status: 'active', metrics: { tasks_completed: 203, tasks_failed: 5, avg_response_time: 176 } },
        { name: 'Analysis', status: 'active', metrics: { tasks_completed: 203, tasks_failed: 5, avg_response_time: 176 } },
        { name: 'Learning', status: 'active', metrics: { tasks_completed: 203, tasks_failed: 5, avg_response_time: 176 } },
        { name: 'Monitor', status: 'active', metrics: { tasks_completed: 203, tasks_failed: 5, avg_response_time: 176 } },
        { name: 'Report', status: 'active', metrics: { tasks_completed: 203, tasks_failed: 5, avg_response_time: 176 } },
    ];

    return (
        <TableContainer component={Box}>
            <Table size="small">
                <TableHead>
                    <TableRow sx={{ '& th': { borderBottom: 'none', color: '#131313', fontWeight: 500, py: 1.5, backgroundColor:'#F3F3F3' } }}>
                        <TableCell>Agent Name</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Tasks Completed</TableCell>
                        <TableCell>Tasks Failed</TableCell>
                        <TableCell>Avg Response Time</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {agentList.map((agent: Agent & { name: string }) => (
                        <TableRow key={agent.name} hover sx={{ '& td': { borderBottom: '1px solid #f1f5f9', py: 1.5 } }}>
                            <TableCell sx={{ fontWeight: 500, color: '#475569' }}>
                                {agent.name.charAt(0).toUpperCase() + agent.name.slice(1)}
                            </TableCell>
                            <TableCell>
                                <Chip
                                    label={agent.status.charAt(0).toUpperCase() + agent.status.slice(1)}
                                    size="small"
                                    sx={{
                                        fontWeight: 700,
                                        fontSize: '0.7rem',
                                        bgcolor: agent.status === 'active' ? '#dcfce7' : '#f1f5f9',
                                        color: agent.status === 'active' ? '#166534' : '#64748b',
                                        borderRadius: '6px'
                                    }}
                                />
                            </TableCell>
                            <TableCell sx={{ color: '#475569', fontWeight: 500 }}>{agent.metrics?.tasks_completed || 0}</TableCell>
                            <TableCell sx={{ color: agent.metrics?.tasks_failed > 5 ? '#ef4444' : '#475569', fontWeight: 700 }}>
                                {agent.metrics?.tasks_failed || (agent.name.toLowerCase() === 'audit' ? 8 : 0)}
                            </TableCell>
                            <TableCell sx={{ color: '#475569', fontWeight: 500 }}>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                    <TimerIcon sx={{ fontSize: 16, color: '#94a3b8' }} />
                                    {Math.round(agent.metrics?.avg_response_time || agent.metrics?.avg_response_time_ms || 245)} ms
                                </Box>
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

const Dashboard: React.FC = () => {
    const [data, setData] = useState<DashboardData | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await api.getStatus();
                setData(response.data as DashboardData);
            } catch (error) {
                console.error('Error fetching dashboard data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
                <CircularProgress />
            </Box>
        );
    }

    const chartData = [
        { name: 'Audit', value: 3.8 },
        { name: 'Testing', value: 3.8 },
        { name: 'Analysis', value: 1.6 },
        { name: 'Learning', value: 6.2 },
        { name: 'Monitoring', value: 1.6 },
        { name: 'Reporting', value: 2.3 },
        { name: 'Remediation', value: 8.5 },
    ];

    return (
        <Box sx={{ p: 1 }}>
            <StatsRow data={data} />

            <Grid container spacing={3}>
                <Grid size={{ xs: 12, lg: 8 }}>
                    <Paper sx={{  height: '100%', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', borderRadius: 3 }}>
                        <Box sx={{ p: 3, }}><Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5, color:'#131313' }}>Agents Overview</Typography>
                        <Typography variant="body2"  sx={{  fontWeight: 400, color:"#666D73" }}>Real-time performance metrics for all agents</Typography>
                        </Box>
                        <AgentsTable agents={data?.agents} />
                    </Paper>
                </Grid>

                <Grid size={{ xs: 12, lg: 4 }}>
                    <Paper sx={{ p: 3, height: '100%', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', borderRadius: 3 }}>
                        <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5, color:'#131313' }}>Agent Distribution</Typography>
                        <Typography variant="body2" sx={{fontWeight: 400, color:"#666D73" }}>Number of agents assigned to each policy type</Typography>
                        <Box sx={{ width: '100%', height: 350, mt: 2 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={chartData} margin={{ top: 0, right: 0, left: -25, bottom: 20 }}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                    <XAxis
                                        dataKey="name"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fontSize: 10, fill: '#94a3b8', fontWeight: 600 }}
                                        interval={0}
                                        angle={-45}
                                        textAnchor="end"
                                    />
                                    <YAxis
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fontSize: 10, fill: '#94a3b8', fontWeight: 600 }}
                                    />
                                    <RechartsTooltip
                                        cursor={{ fill: '#f8fafc' }}
                                        contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                                    />
                                    <Bar
                                        dataKey="value"
                                        fill="#6366f1"
                                        radius={[6, 6, 0, 0]}
                                        barSize={32}
                                    >
                                        {chartData.map((_entry, index) => (
                                            <Cell key={`cell-${index}`} fill="#6366f1" />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </Box>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default Dashboard;
