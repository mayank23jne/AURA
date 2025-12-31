import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Button,
    Modal,
    IconButton,
    Grid
} from '@mui/material';
import {
    Close as CloseIcon,
    Memory as MemoryIcon,
    Timer as UptimeIcon,
    Favorite as HeartbeatIcon,
    AccessTime as TimeIcon,
    Visibility as EyeIcon
} from '@mui/icons-material';
import { api } from '../api/client';

const Agent: React.FC = () => {
    const [agents, setAgents] = useState<any[]>([]);
    const [selectedAgent, setSelectedAgent] = useState<any | null>(null);
    const [agentMetrics, setAgentMetrics] = useState<any | null>(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [loadingMetrics, setLoadingMetrics] = useState(false);

    useEffect(() => {
        const fetchAgents = async () => {
            try {
                const response = await api.getAgents();
                setAgents(response.data.agents);
            } catch (error) {
                console.error('Error fetching agents:', error);
            }
        };
        fetchAgents();
    }, []);

    const handleOpenModal = async (agent: any) => {
        setSelectedAgent(agent);
        setIsModalOpen(true);
        setLoadingMetrics(true);
        try {
            const response = await api.getAgentMetrics(agent.name);
            setAgentMetrics(response.data);
        } catch (error) {
            console.error('Error fetching agent metrics:', error);
        } finally {
            setLoadingMetrics(false);
        }
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setSelectedAgent(null);
        setAgentMetrics(null);
    };

    return (
        <Box sx={{ mt: 2 }}>
            <Paper sx={{ height: '100%', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', borderRadius: 2 }}>
                <Box sx={{ p: 2, }}>
                    <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5, color: '#131313' }}>Agent List</Typography>
                    <Typography variant="body2" sx={{ fontWeight: 400, color: '#666D73', fontSize: '16px' }}>
                        {agents.length} agents registered
                    </Typography>
                </Box>

                <TableContainer component={Paper} sx={{ borderRadius: '0 0 8px 8px', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
                    <Table>
                        <TableHead>
                            <TableRow sx={{ '& th': { borderBottom: 'none', color: '#131313', fontWeight: 500, py: 1, backgroundColor: '#F3F3F3' } }}>
                                <TableCell>Agent Name</TableCell>
                                <TableCell>ID</TableCell>
                                <TableCell>Status</TableCell>
                                <TableCell align="right">Tasks Completed</TableCell>
                                <TableCell align="right">Tasks Failed</TableCell>
                                <TableCell>Avg Response Time</TableCell>
                                <TableCell align="center">Actions</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {agents.map((agent, index) => {
                                const agentName = agent.name.charAt(0).toUpperCase() + agent.name.slice(1);
                                const tasksCompleted = Math.floor(Math.random() * 150) + 50;
                                const tasksFailed = Math.floor(Math.random() * 50) + 10;
                                const avgResponseTime = Math.floor(Math.random() * 200) + 100;

                                return (
                                    <TableRow
                                        key={agent.id}
                                        hover
                                        sx={{
                                            '&:last-child td, &:last-child th': { border: 0 },
                                            '& td': { py: 1 }
                                        }}
                                    >
                                        <TableCell sx={{ fontWeight: 500, color: '#666D73', fontSize: '14px' }}>
                                            Agent {agentName}
                                        </TableCell>
                                        <TableCell sx={{
                                            color: index === 2 ? '#EF4444' : '#666D73',
                                            fontFamily: 'Lexend',
                                            fontSize: '0.875rem'
                                        }}>
                                            {agent.id}
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                label={agent.status === 'active' || agent.status === 'idle' ? 'Active' : 'Inactive'}
                                                size="small"
                                                sx={{
                                                    fontWeight: 500,
                                                    fontSize: '0.75rem',
                                                    bgcolor: agent.status === 'active' || agent.status === 'idle' ? '#DCFCE7' : '#FEE2E2',
                                                    color: agent.status === 'active' || agent.status === 'idle' ? '#166534' : '#991B1B',
                                                    borderRadius: '6px',
                                                    height: '24px'
                                                }}
                                            />
                                        </TableCell>
                                        <TableCell align="right" sx={{ fontWeight: 500, color: '#666D73' }}>
                                            {tasksCompleted}
                                        </TableCell>
                                        <TableCell align="right" sx={{ fontWeight: 500, color: '#666D73' }}>
                                            {tasksFailed}
                                        </TableCell>
                                        <TableCell>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                <TimeIcon sx={{ fontSize: '1rem', color: '#666D73' }} />
                                                <Typography variant="body2" sx={{ fontWeight: 500, color: '#666D73' }}>
                                                    {avgResponseTime} ms
                                                </Typography>
                                            </Box>
                                        </TableCell>
                                        <TableCell align="center">
                                            <Button
                                                variant="text"
                                                size="small"
                                                onClick={() => handleOpenModal(agent)}
                                                startIcon={<EyeIcon sx={{ fontSize: '1rem' }} />}
                                                sx={{
                                                    textTransform: 'none',
                                                    fontWeight: 500,
                                                    color: '#6366F1',
                                                    fontSize: '0.875rem',
                                                    border: 'none',
                                                    '&:hover': {
                                                        bgcolor: 'rgba(99, 102, 241, 0.04)',
                                                        border: 'none'
                                                    }
                                                }}
                                            >
                                                View More
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                );
                            })}
                        </TableBody>
                    </Table>
                </TableContainer>

                {/* Agent Detail Modal */}
                <Modal
                    open={isModalOpen}
                    onClose={handleCloseModal}
                    aria-labelledby="agent-detail-modal"

                >
                    <Box sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        width: 800,
                        maxHeight: '90vh',
                        bgcolor: 'background.paper',
                        borderRadius: 3,
                        boxShadow: 24,
                        p: 0,
                        border: 'none',
                        overflow: 'auto'
                    }}>
                        {/* Header */}
                        <Box sx={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            p: 2,
                            borderBottom: '1px solid #E8E8E8'
                        }}>
                            <Box>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                    <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313', fontSize: '20px' }}>
                                        {selectedAgent?.name.charAt(0).toUpperCase() + selectedAgent?.name.slice(1)}
                                    </Typography>
                                    <Chip
                                        label={selectedAgent?.status === 'active' || selectedAgent?.status === 'idle' ? 'Active' : 'Inactive'}
                                        size="small"
                                        sx={{
                                            fontWeight: 500,
                                            fontSize: '0.75rem',
                                            bgcolor: selectedAgent?.status === 'active' || selectedAgent?.status === 'idle' ? '#DCFCE7' : '#FEE2E2',
                                            color: selectedAgent?.status === 'active' || selectedAgent?.status === 'idle' ? '#166534' : '#991B1B',
                                            borderRadius: '6px',
                                            height: '24px'
                                        }}
                                    />
                                </Box>
                                <Box >
                                    <Typography variant="body2" sx={{ color: '#666D73', fontWeight: 400, fontSize: '12px' }}>
                                        Agent Metrics & Performance
                                    </Typography>
                                </Box>
                            </Box>

                            <IconButton onClick={handleCloseModal} size="small">
                                <CloseIcon sx={{ backgroundColor: '#595BFF', color: '#FFFFFF', borderRadius: 1 }} />
                            </IconButton>
                        </Box>

                        {/* Subtitle */}


                        {/* Metrics Grid */}
                        <Box sx={{ p: 3, pt: 2 }}>
                            <Grid container spacing={2}>
                                {/* Tasks Completed */}
                                <Grid size={{ xs: 4 }}>
                                    <Paper sx={{
                                        display: 'flex', alignItems: 'center', gap: 2,
                                        p: 2,
                                        borderRadius: 2,
                                        border: '1px solid #E8E8E8',
                                        boxShadow: 'none'
                                    }}>
                                        <Box sx={{ mb: 1 }}>

                                            <img src="./TaskIcon (1).png" />

                                        </Box>
                                        <Box>
                                            <Typography variant="caption" sx={{ color: '#666D73', fontWeight: 400, display: 'block', }}>
                                                Tasks Completed
                                            </Typography>
                                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313' }}>
                                                {loadingMetrics ? '...' : agentMetrics?.tasks_completed || 12}
                                            </Typography>
                                        </Box>
                                    </Paper>
                                </Grid>

                                {/* Tasks Failed */}
                                <Grid size={{ xs: 4 }}>
                                    <Paper sx={{
                                        display: 'flex', alignItems: 'center', gap: 2,
                                        p: 2,
                                        borderRadius: 2,
                                        border: '1px solid #E8E8E8',
                                        boxShadow: 'none'
                                    }}>
                                        <Box sx={{ mb: 0.5 }}>

                                            <img src="./failedIcon (1).png" />

                                        </Box>
                                        <Box>
                                            <Typography variant="caption" sx={{ color: '#666D73', fontWeight: 400, display: 'block', }}>
                                                Tasks Failed
                                            </Typography>
                                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313' }}>
                                                {loadingMetrics ? '...' : agentMetrics?.tasks_failed || 5}
                                            </Typography>
                                        </Box>
                                    </Paper>
                                </Grid>

                                {/* Avg Response Time */}
                                <Grid size={{ xs: 4 }}>
                                    <Paper sx={{
                                        display: 'flex', alignItems: 'center', gap: 2,
                                        p: 2,
                                        borderRadius: 2,
                                        border: '1px solid #E8E8E8',
                                        boxShadow: 'none'
                                    }}>
                                        <Box sx={{ mb: 0.5 }}>

                                            <img src="./AvgIcon (1).png" />

                                        </Box>
                                        <Box>
                                            <Typography variant="caption" sx={{ color: '#666D73', fontWeight: 400, display: 'block', mb: 0.5 }}>
                                                Avg Response Time
                                            </Typography>
                                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313' }}>
                                                {loadingMetrics ? '...' : agentMetrics?.avg_response_time_ms || 12}
                                            </Typography>
                                        </Box>
                                    </Paper>
                                </Grid>

                                {/* Memory Usage */}
                                <Grid size={{ xs: 4 }}>
                                    <Paper sx={{
                                        display: 'flex', alignItems: 'center', gap: 2,
                                        p: 2,
                                        borderRadius: 2,
                                        border: '1px solid #E8E8E8',
                                        boxShadow: 'none'
                                    }}>
                                        <Box sx={{ mb: 0.5 }}>

                                            <img src="./memoryIcon (1).png" />

                                        </Box>
                                        <Box>
                                            <Typography variant="caption" sx={{ color: '#666D73', fontWeight: 400, display: 'block', mb: 0.5 }}>
                                                Memory Usage
                                            </Typography>
                                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313' }}>
                                                {loadingMetrics ? '...' : `${Math.round(agentMetrics?.memory_usage_mb || 12)} MB`}
                                            </Typography>
                                        </Box>
                                    </Paper>
                                </Grid>

                                {/* Uptime */}
                                <Grid size={{ xs: 4 }}>
                                    <Paper sx={{
                                        display: 'flex', alignItems: 'center', gap: 2,
                                        p: 2,
                                        borderRadius: 2,
                                        border: '1px solid #E8E8E8',
                                        boxShadow: 'none'
                                    }}>
                                        <Box sx={{ mb: 0.5 }}>

                                            <img src="./uptimeIcon (1).png" />

                                        </Box>
                                        <Box>
                                            <Typography variant="caption" sx={{ color: '#666D73', fontWeight: 400, display: 'block', mb: 0.5 }}>
                                                Uptime
                                            </Typography>
                                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313' }}>
                                                {loadingMetrics ? '...' : `0.2 HRS`}
                                            </Typography>
                                        </Box>
                                    </Paper>
                                </Grid>

                                {/* Last Heartbeat */}
                                <Grid size={{ xs: 4 }}>
                                    <Paper sx={{
                                        display: 'flex', alignItems: 'center', gap: 2,
                                        p: 2,
                                        borderRadius: 2,
                                        border: '1px solid #E8E8E8',
                                        boxShadow: 'none'
                                    }}>
                                        <Box sx={{ mb: 0.5 }}>

                                            <img src="./lastheartIcon (1).png" />

                                        </Box>
                                        <Box>
                                            <Typography variant="caption" sx={{ color: '#666D73', fontWeight: 400, display: 'block', mb: 0.5 }}>
                                                Last Heartbeat
                                            </Typography>
                                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313' }}>
                                                02/12/2025
                                            </Typography>
                                        </Box>
                                    </Paper>
                                </Grid>
                            </Grid>

                            {/* Raw Metrics Data */}
                            <Box sx={{ mt: 3, backgroundColor: '#D9D9D9', borderRadius: 2, }}>
                                <Typography variant="subtitle2" sx={{ pl: 1, pt: 1, mb: 1.5, fontWeight: 600, color: '#131313' }}>
                                    Raw Metrics Data
                                </Typography>
                                <Paper sx={{
                                    p: 2.5,
                                    bgcolor: '#F8F9FA',
                                    borderRadius: 0,
                                    border: '1px solid #E8E8E8',
                                    boxShadow: 'none'
                                }}>
                                    <pre style={{
                                        margin: 0,
                                        overflow: 'auto',
                                        fontFamily: 'Lexend, monospace',
                                        fontSize: '0.75rem',
                                        color: '#131313',
                                        lineHeight: 1.6
                                    }}>
                                        {loadingMetrics ? 'Loading metrics...' : JSON.stringify({ ...selectedAgent, ...agentMetrics }, null, 2)}
                                    </pre>
                                </Paper>
                            </Box>
                        </Box>
                    </Box>
                </Modal>
            </Paper>
        </Box>
    );
};

export default Agent;
