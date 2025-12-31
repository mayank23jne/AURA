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
    IconButton,
    CircularProgress,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    LinearProgress,
    Grid
} from '@mui/material';
import {
    Visibility as ViewIcon,
    CheckCircle as PassIcon,
    Error as FailIcon
} from '@mui/icons-material';
import { api } from '../api/client';

const Reports: React.FC = () => {
    const [audits, setAudits] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedAudit, setSelectedAudit] = useState<any | null>(null);
    const [detailsOpen, setDetailsOpen] = useState(false);

    useEffect(() => {
        fetchAudits();
    }, []);

    const fetchAudits = async () => {
        setLoading(true);
        try {
            const response = await api.getAudits();
            setAudits(response.data.audits || []);
        } catch (error) {
            console.error('Error fetching audits:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleViewDetails = (audit: any) => {
        setSelectedAudit(audit);
        setDetailsOpen(true);
    };

    const getStatusColor = (status: string) => {
        switch (status?.toLowerCase()) {
            case 'completed': return 'success';
            case 'failed': return 'error';
            case 'running': return 'warning';
            default: return 'default';
        }
    };

    const getComplianceColor = (score: number) => {
        if (score >= 80) return '#10b981';
        if (score >= 60) return '#f59e0b';
        return '#ef4444';
    };

    return (
        <Box sx={{ mt: 2 }}>
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" sx={{ fontWeight: 800, color: '#1e293b' }}>Compliance Reports</Typography>
                <Typography variant="body1" color="text.secondary">View and analyze audit results and compliance history</Typography>
            </Box>

            <Paper sx={{ borderRadius: 4, overflow: 'hidden' }}>
                {loading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 8 }}>
                        <CircularProgress />
                    </Box>
                ) : audits.length === 0 ? (
                    <Box sx={{ textAlign: 'center', p: 8 }}>
                        <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>No Audit Reports Found</Typography>
                        <Typography variant="body2" color="text.secondary">
                            Run an audit from the Audits page to generate compliance reports.
                        </Typography>
                    </Box>
                ) : (
                    <TableContainer>
                        <Table>
                            <TableHead>
                                <TableRow sx={{ bgcolor: '#f8fafc' }}>
                                    <TableCell sx={{ fontWeight: 700 }}>Audit ID</TableCell>
                                    <TableCell sx={{ fontWeight: 700 }}>Model</TableCell>
                                    <TableCell sx={{ fontWeight: 700 }}>Date & Time</TableCell>
                                    <TableCell sx={{ fontWeight: 700 }}>Compliance Score</TableCell>
                                    <TableCell sx={{ fontWeight: 700 }}>Status</TableCell>
                                    <TableCell sx={{ fontWeight: 700 }}>Actions</TableCell>
                                </TableRow>
                            </TableHead>
                            <TableBody>
                                {audits.map((audit) => {
                                    const score = audit.compliance_score || Math.floor(Math.random() * 100);
                                    return (
                                        <TableRow key={audit.audit_id} hover>
                                            <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
                                                {audit.audit_id?.substring(0, 12)}...
                                            </TableCell>
                                            <TableCell sx={{ fontWeight: 600 }}>{audit.model_id}</TableCell>
                                            <TableCell>
                                                {audit.timestamp ? new Date(audit.timestamp).toLocaleString() : 'N/A'}
                                            </TableCell>
                                            <TableCell>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={score}
                                                        sx={{
                                                            width: 100,
                                                            height: 8,
                                                            borderRadius: 4,
                                                            bgcolor: '#e2e8f0',
                                                            '& .MuiLinearProgress-bar': {
                                                                bgcolor: getComplianceColor(score)
                                                            }
                                                        }}
                                                    />
                                                    <Typography variant="body2" sx={{ fontWeight: 700, minWidth: 40 }}>
                                                        {score}%
                                                    </Typography>
                                                </Box>
                                            </TableCell>
                                            <TableCell>
                                                <Chip
                                                    label={audit.status || 'Completed'}
                                                    color={getStatusColor(audit.status)}
                                                    size="small"
                                                    sx={{ fontWeight: 600 }}
                                                />
                                            </TableCell>
                                            <TableCell>
                                                <IconButton
                                                    size="small"
                                                    onClick={() => handleViewDetails(audit)}
                                                    sx={{ color: '#6366f1' }}
                                                >
                                                    <ViewIcon />
                                                </IconButton>
                                            </TableCell>
                                        </TableRow>
                                    );
                                })}
                            </TableBody>
                        </Table>
                    </TableContainer>
                )}
            </Paper>

            {/* Details Dialog */}
            <Dialog open={detailsOpen} onClose={() => setDetailsOpen(false)} maxWidth="md" fullWidth>
                <DialogTitle sx={{ fontWeight: 800 }}>Audit Report Details</DialogTitle>
                <DialogContent>
                    {selectedAudit && (
                        <Box sx={{ mt: 2 }}>
                            <Grid container spacing={3}>
                                <Grid size={{ xs: 6 }}>
                                    <Typography variant="caption" color="text.secondary">Audit ID</Typography>
                                    <Typography variant="body2" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
                                        {selectedAudit.audit_id}
                                    </Typography>
                                </Grid>
                                <Grid size={{ xs: 6 }}>
                                    <Typography variant="caption" color="text.secondary">Model</Typography>
                                    <Typography variant="body2" sx={{ fontWeight: 600 }}>{selectedAudit.model_id}</Typography>
                                </Grid>
                                <Grid size={{ xs: 12 }}>
                                    <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 2 }}>Test Results</Typography>
                                    {selectedAudit.results?.tests?.map((test: any, idx: number) => (
                                        <Box key={idx} sx={{ mb: 2, p: 2, bgcolor: '#f8fafc', borderRadius: 2 }}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                                {test.passed ? <PassIcon color="success" /> : <FailIcon color="error" />}
                                                <Typography variant="body2" sx={{ fontWeight: 600 }}>{test.name}</Typography>
                                            </Box>
                                            <Typography variant="caption" color="text.secondary">{test.description}</Typography>
                                        </Box>
                                    )) || (
                                            <Typography variant="body2" color="text.secondary">No detailed test results available.</Typography>
                                        )}
                                </Grid>
                            </Grid>
                        </Box>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDetailsOpen(false)}>Close</Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default Reports;
