import React, { useState, useEffect } from 'react';
import {
    Box,
    Paper,
    Typography,
    Button,
    TextField,
    Grid,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    CircularProgress,
    alpha
} from '@mui/material';
import {
    Add as AddIcon,
    Edit as EditIcon,
    DeleteOutline as DeleteIcon,
    Refresh as RefreshIcon,
    Search as SearchIcon,
    Archive as ArchiveIcon
} from '@mui/icons-material';
import { api } from '../api/client';

// Interfaces
interface Package {
    id: string;
    name: string;
    description: string;
    created_at: string;
    policies_count: number;
}

const Packages: React.FC = () => {
    // State
    const [packages, setPackages] = useState<Package[]>([]);
    const [loading, setLoading] = useState(false);
    const [actionLoading, setActionLoading] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');

    // Create Modal State
    const [createDialogOpen, setCreateDialogOpen] = useState(false);
    const [newPackage, setNewPackage] = useState({ name: '', description: '' });

    // Edit Modal State
    const [editDialogOpen, setEditDialogOpen] = useState(false);
    const [editingPackage, setEditingPackage] = useState<Package | null>(null);
    const [editForm, setEditForm] = useState({ name: '', description: '' });

    // Delete Modal State
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [packageToDelete, setPackageToDelete] = useState<Package | null>(null);

    // Initial Fetch
    useEffect(() => {
        fetchPackages();
    }, []);

    const fetchPackages = async () => {
        setLoading(true);
        try {
            const response = await api.getPackages();
            setPackages(response.data.packages || []);
        } catch (error) {
            console.error('Error fetching packages:', error);
        } finally {
            setLoading(false);
        }
    };

    // Create Handler
    const handleCreate = async () => {
        if ( !newPackage.name) return;
        setActionLoading(true);
        try {
            await api.createPackage(newPackage);
            setCreateDialogOpen(false);
            setNewPackage({ name: '', description: '' });
            fetchPackages();
        } catch (error) {
            console.error('Error creating package:', error);
            alert('Failed to create package');
        } finally {
            setActionLoading(false);
        }
    };

    // Edit Handlers
    const openEditDialog = (pkg: Package) => {
        setEditingPackage(pkg);
        setEditForm({ name: pkg.name, description: pkg.description });
        setEditDialogOpen(true);
    };

    const handleUpdate = async () => {
        if (!editingPackage) return;
        setActionLoading(true);
        try {
            await api.updatePackage(editingPackage.id, editForm);
            setEditDialogOpen(false);
            fetchPackages();
        } catch (error) {
            console.error('Error updating package:', error);
            alert('Failed to update package');
        } finally {
            setActionLoading(false);
        }
    };

    // Delete Handlers
    const openDeleteDialog = (pkg: Package) => {
        setPackageToDelete(pkg);
        setDeleteDialogOpen(true);
    };

    const handleDelete = async () => {
        if (!packageToDelete) return;
        setActionLoading(true);
        try {
            await api.deletePackage(packageToDelete.id);
            setDeleteDialogOpen(false);
            fetchPackages();
        } catch (error) {
            console.error('Error deleting package:', error);
            alert('Failed to delete package');
        } finally {
            setActionLoading(false);
        }
    };

    // Filter Logic
    const filteredPackages = packages.filter(p =>
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.id.toLowerCase().includes(searchQuery.toLowerCase())
    );

    return (
        <Box sx={{
            mx: -2,
            mb: -2,
            width: 'calc(100% + 32px)',
            minHeight: '100%',
            overflowX: 'hidden',
            bgcolor: '#fbfbfb',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
        }}>
            {/* Header Section */}
            <Box sx={{
                width: '100%',
                bgcolor: '#fff',
                borderBottom: '1px solid #E8E8E8',
                px: 4,
                py: 2,
                mt: -2,
                display: 'flex',
                justifyContent: 'center'
            }}>
                <Box sx={{ width: '100%', maxWidth: '1440px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Box sx={{
                            p: 1.5,
                            borderRadius: '12px',
                            bgcolor: alpha('#6366f1', 0.1),
                            color: '#6366f1',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <ArchiveIcon sx={{ fontSize: 24 }} />
                        </Box>
                        <Box>
                            <Typography variant="h6" sx={{ fontWeight: 600, color: '#131313', lineHeight: 1.2 }}>
                                Policy Packages
                            </Typography>
                            <Typography variant="body2" sx={{ color: '#666D73' }}>
                                Manage collections of policies for standardized auditing.
                            </Typography>
                        </Box>
                    </Box>

                    <Button
                        variant="contained"
                        startIcon={<AddIcon />}
                        onClick={() => setCreateDialogOpen(true)}
                        sx={{
                            textTransform: 'none',
                            bgcolor: '#6366f1',
                            borderRadius: '8px',
                            boxShadow: 'none',
                            px: 3,
                            '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' }
                        }}
                    >
                        Create Package
                    </Button>
                </Box>
            </Box>

            {/* Content Section */}
            <Box sx={{ width: '100%', maxWidth: '1440px', px: 4, mt: 4 }}>
                <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <TextField
                        size="small"
                        placeholder="Search packages..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        InputProps={{
                            startAdornment: <SearchIcon sx={{ color: '#94A3B8', mr: 1, fontSize: 20 }} />
                        }}
                        sx={{
                            width: 320,
                            bgcolor: '#fff',
                            '& .MuiOutlinedInput-root': { borderRadius: 2 }
                        }}
                    />
                    <IconButton onClick={fetchPackages} sx={{ color: '#6366f1', bgcolor: alpha('#6366f1', 0.1) }}>
                        <RefreshIcon sx={{ fontSize: 20 }} />
                    </IconButton>
                </Box>

                <Paper sx={{ borderRadius: 3, overflow: 'hidden', border: '1px solid #E8E8E8', boxShadow: 'none' }}>
                    {/* Table Header */}
                    <Box sx={{
                        display: 'grid',
                        gridTemplateColumns: '1.5fr 2fr 3fr 1fr 100px',
                        px: 3,
                        py: 2,
                        bgcolor: '#f8fafc',
                        borderBottom: '1px solid #E8E8E8'
                    }}>
                        {['ID', 'Name', 'Description', 'Policies', 'Actions'].map((h, i) => (
                            <Typography key={i} sx={{
                                fontSize: '0.75rem',
                                fontWeight: 600,
                                color: '#64748b',
                                textTransform: 'uppercase',
                                textAlign: i === 4 ? 'right' : 'left'
                            }}>
                                {h}
                            </Typography>
                        ))}
                    </Box>

                    {/* Table Body */}
                    {loading ? (
                        <Box sx={{ p: 4, textAlign: 'center' }}><CircularProgress /></Box>
                    ) : (
                        <Box>
                            {filteredPackages.length > 0 ? filteredPackages.map((pkg, index) => (
                                <Box key={pkg.id} sx={{
                                    display: 'grid',
                                    gridTemplateColumns: '1.5fr 2fr 3fr 1fr 100px',
                                    px: 3,
                                    py: 2.5,
                                    borderBottom: index === filteredPackages.length - 1 ? 'none' : '1px solid #E8E8E8',
                                    alignItems: 'center',
                                    '&:hover': { bgcolor: '#fbfbfb' }
                                }}>
                                    <Typography sx={{ fontSize: '0.85rem', fontWeight: 500, color: '#334155' }}>{pkg.id}</Typography>
                                    <Typography sx={{ fontSize: '0.9rem', fontWeight: 600, color: '#0f172a' }}>{pkg.name}</Typography>
                                    <Typography sx={{ fontSize: '0.85rem', color: '#64748b', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', pr: 2 }}>
                                        {pkg.description || '-'}
                                    </Typography>
                                    <Box>
                                        <Box sx={{
                                            display: 'inline-flex',
                                            bgcolor: alpha('#6366f1', 0.1),
                                            color: '#6366f1',
                                            px: 1.5,
                                            py: 0.5,
                                            borderRadius: 2,
                                            fontSize: '0.75rem',
                                            fontWeight: 600
                                        }}>
                                            {pkg.policies_count} Policies
                                        </Box>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
                                        <IconButton size="small" onClick={() => openEditDialog(pkg)} sx={{ color: '#64748b', '&:hover': { color: '#6366f1' } }}>
                                            <EditIcon sx={{ fontSize: 18 }} />
                                        </IconButton>
                                        <IconButton size="small" onClick={() => openDeleteDialog(pkg)} sx={{ color: '#ef4444', '&:hover': { bgcolor: alpha('#ef4444', 0.1) } }}>
                                            <DeleteIcon sx={{ fontSize: 18 }} />
                                        </IconButton>
                                    </Box>
                                </Box>
                            )) : (
                                <Box sx={{ p: 6, textAlign: 'center' }}>
                                    <Typography sx={{ color: '#64748b' }}>No packages found.</Typography>
                                </Box>
                            )}
                        </Box>
                    )}
                </Paper>
            </Box>

            {/* Create Dialog */}
            <Dialog open={createDialogOpen} onClose={() => setCreateDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{ borderBottom: '1px solid #E8E8E8' }}>Create New Package</DialogTitle>
                <DialogContent sx={{ pt: 3 }}>
                    <Grid container spacing={3} sx={{ mt: 0 }}>
                        {/* <Grid item xs={12}>
                            <Typography sx={{ mb: 1, fontWeight: 500, fontSize: '0.9rem' }}>Package ID</Typography>
                            <TextField
                                fullWidth size="small" placeholder="e.g., finance-audit-pack"
                                value={newPackage.id}
                                onChange={(e) => setNewPackage({ ...newPackage, id: e.target.value })}
                            />
                        </Grid> */}
                        <Grid item xs={12}>
                            <Typography sx={{ mb: 1, fontWeight: 500, fontSize: '0.9rem' }}>Package Name</Typography>
                            <TextField
                                fullWidth size="small" placeholder="Display Name"
                                value={newPackage.name}
                                onChange={(e) => setNewPackage({ ...newPackage, name: e.target.value })}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Typography sx={{ mb: 1, fontWeight: 500, fontSize: '0.9rem' }}>Description</Typography>
                            <TextField
                                fullWidth multiline rows={3} size="small" placeholder="Describe the purpose of this package"
                                value={newPackage.description}
                                onChange={(e) => setNewPackage({ ...newPackage, description: e.target.value })}
                            />
                        </Grid>
                    </Grid>
                </DialogContent>
                <DialogActions sx={{ p: 2.5, borderTop: '1px solid #E8E8E8' }}>
                    <Button onClick={() => setCreateDialogOpen(false)} sx={{ color: '#64748b' }}>Cancel</Button>
                    <Button
                        variant="contained"
                        onClick={handleCreate}
                        disabled={actionLoading || !newPackage.name}
                        sx={{ bgcolor: '#6366f1', boxShadow: 'none' }}
                    >
                        {actionLoading ? <CircularProgress size={20} color="inherit" /> : 'Create Package'}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Edit Dialog */}
            <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{ borderBottom: '1px solid #E8E8E8' }}>Edit Package</DialogTitle>
                <DialogContent sx={{ pt: 3 }}>
                    <Grid container spacing={3} sx={{ mt: 0 }}>
                        <Grid item xs={12}>
                            <Typography sx={{ mb: 1, fontWeight: 500, fontSize: '0.9rem' }}>Package Name</Typography>
                            <TextField
                                fullWidth size="small"
                                value={editForm.name}
                                onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                            />
                        </Grid>
                        <Grid item xs={12}>
                            <Typography sx={{ mb: 1, fontWeight: 500, fontSize: '0.9rem' }}>Description</Typography>
                            <TextField
                                fullWidth multiline rows={3} size="small"
                                value={editForm.description}
                                onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
                            />
                        </Grid>
                    </Grid>
                </DialogContent>
                <DialogActions sx={{ p: 2.5, borderTop: '1px solid #E8E8E8' }}>
                    <Button onClick={() => setEditDialogOpen(false)} sx={{ color: '#64748b' }}>Cancel</Button>
                    <Button
                        variant="contained"
                        onClick={handleUpdate}
                        disabled={actionLoading}
                        sx={{ bgcolor: '#6366f1', boxShadow: 'none' }}
                    >
                        {actionLoading ? <CircularProgress size={20} color="inherit" /> : 'Save Changes'}
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Delete Dialog */}
            <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
                <DialogTitle>Confirm Deletion</DialogTitle>
                <DialogContent>
                    <Typography>
                        Are you sure you want to delete the package <strong>{packageToDelete?.name}</strong>?
                        This will unlink all associated policies but will not delete them.
                    </Typography>
                </DialogContent>
                <DialogActions sx={{ p: 2 }}>
                    <Button onClick={() => setDeleteDialogOpen(false)} sx={{ color: '#64748b' }}>Cancel</Button>
                    <Button
                        variant="contained"
                        color="error"
                        onClick={handleDelete}
                        disabled={actionLoading}
                        sx={{ boxShadow: 'none' }}
                    >
                        {actionLoading ? <CircularProgress size={20} color="inherit" /> : 'Delete Package'}
                    </Button>
                </DialogActions>
            </Dialog>

        </Box>
    );
};

export default Packages;
