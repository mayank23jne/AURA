import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Paper,
    TextField,
    Button,
    Tabs,
    Tab,
    alpha,
    CircularProgress,
    Grid,
    MenuItem,
    InputAdornment,
    Checkbox,
    FormControlLabel,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    IconButton
} from '@mui/material';
import {
    CloudUploadOutlined,
    SearchOutlined,
    EditOutlined,
    DeleteOutline,
    RefreshOutlined,
    CloseOutlined,
    SaveOutlined
} from '@mui/icons-material';
import { api } from '../api/client';
import { styled } from '@mui/material/styles';
import type { Policy } from '../Types';

const IconButtonSmall = styled(IconButton)({
    padding: 4,
    borderRadius: 6,
    border: '1px solid #f1f5f9',
    '&:hover': {
        backgroundColor: '#f8fafc',
    }
});

const Policy: React.FC = () => {
    const [policies, setPolicies] = useState<Policy[]>([]);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [activeTab, setActiveTab] = useState(0);

    // Dialog state
    const [editDialogOpen, setEditDialogOpen] = useState(false);
    const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
    const [editingPolicy, setEditingPolicy] = useState<Policy | null>(null);
    const [policyToDelete, setPolicyToDelete] = useState<Policy | null>(null);
    const [editForm, setEditForm] = useState<any>({
        name: '',
        description: '',
        category: '',
        severity: '',
        packageId: '',
        version: '',
        active: true,
        rules: '',
        references: ''
    });

    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);

    // Form state
    const [regulationName, setRegulationName] = useState('EU AI Act');
    const [regulatoryText, setRegulatoryText] = useState('AI systems must be transparent about their capabilities and limitations. Users must be informed when they are interacting with an AI system. AI systems must not manipulate users through subliminal techniques. High-risk AI systems must undergo conformity assessments...');

    // Manual Policy Form state
    const [manualPolicy, setManualPolicy] = useState({
        policyId: '',
        version: '',
        name: '',
        category: '',
        severity: '',
        packageId: '',
        description: '',
        rules: '',
        references: '',
        active: true
    });

    const handleManualChange = (field: string, value: string) => {
        setManualPolicy(prev => ({ ...prev, [field]: value }));
    };

    const fetchPolicies = async () => {
        setLoading(true);
        try {
            const response = await api.getPolicies();
            setPolicies(response.data.policies);
        } catch (error) {
            console.error('Error fetching policies:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPolicies();
        fetchPackages();
    }, []);

    const [packages, setPackages] = useState<any[]>([]);
    const fetchPackages = async () => {
        try {
            const response = await api.getPackages();
            setPackages(response.data.packages || []);
        } catch (error) {
            console.error('Error fetching packages:', error);
        }
    };

    const handleGenerate = async () => {
        if (!regulatoryText) return;
        setGenerating(true);
        try {
            await api.generatePolicy({
                regulation_text: regulatoryText,
                regulation_name: regulationName
            });
            alert('Policies generated and added successfully!');
            fetchPolicies();
            setActiveTab(3); // Go to Existing Policies
        } catch (error) {
            console.error('Generation failed:', error);
            alert('Failed to generate policies.');
        } finally {
            setGenerating(false);
        }
    };

    type RuleSeverity = 'critical' | 'high' | 'medium' | 'low';

    interface PolicyRule {
        id: string;
        text: string;
        severity: RuleSeverity;
    }

    const parseRules = (
        rulesText: string,
        policyId: string
    ): PolicyRule[] => {
        const rules: PolicyRule[] = [];

        if (rulesText && rulesText.trim()) {
            rulesText
                .trim()
                .split('\n')
                .forEach((line: string, index: number) => {
                    let ruleText = '';
                    let severity: RuleSeverity = 'medium';

                    if (line.includes('|')) {
                        const parts: string[] = line.split('|');
                        ruleText = parts[0].trim();
                        severity = (parts[1]?.trim().toLowerCase() as RuleSeverity) || 'medium';
                    } else {
                        ruleText = line.trim();
                    }

                    if (ruleText) {
                        rules.push({
                            id: `${policyId}-rule-${index + 1}`,
                            text: ruleText,
                            severity
                        });
                    }
                });
        }

        return rules;
    };
    const handleCreateManual = async () => {
        if (!manualPolicy.name || !manualPolicy.policyId) return;
        setGenerating(true);
        try {
            const parsedRules = parseRules(manualPolicy.rules, manualPolicy.policyId);
            await api.createPolicy({
                id: manualPolicy.policyId,
                name: manualPolicy.name,
                description: manualPolicy.description,
                category: manualPolicy.category,
                severity: manualPolicy.severity,
                version: manualPolicy.version,
                active: manualPolicy.active,
                package_id: manualPolicy.packageId || null,
                rules: parsedRules,
                regulatory_references: manualPolicy.references ? [manualPolicy.references] : []
            });
            alert('Policy created successfully!');
            fetchPolicies();
            setActiveTab(3);
        } catch (error) {
            console.error('Creation failed:', error);
            alert('Failed to create policy.');
        } finally {
            setGenerating(false);
        }
    };

    const handleEditClick = (policy: Policy) => {
        setEditingPolicy(policy);
        setEditForm({
            name: policy.name,
            description: policy.description,
            category: policy.category,
            severity: policy.severity || 'medium',
            version: policy.version || '1.0.0',
            packageId: policy.package_id || '',
            active: policy.active,
            rules: (policy.rules || []).map((r: any) => `${r.text} | ${r.severity}`).join('\n'),
            references: (policy.regulatory_references || []).join(', ')
        });
        setEditDialogOpen(true);
    };

    const handleUpdatePolicy = async () => {
        if (!editingPolicy) return;
        setGenerating(true);
        try {
            const parsedRules = parseRules(editForm.rules, editingPolicy.id);
            await api.updatePolicy(editingPolicy.id, {
                name: editForm.name,
                description: editForm.description,
                category: editForm.category,
                severity: editForm.severity,
                version: editForm.version,
                package_id: editForm.packageId || null,
                active: editForm.active,
                rules: parsedRules,
                regulatory_references: editForm.references ? editForm.references.split(',').map((r: string) => r.trim()) : []
            });
            alert('Policy updated successfully!');
            setEditDialogOpen(false);
            fetchPolicies();
        } catch (error) {
            console.error('Update failed:', error);
            alert('Failed to update policy.');
        } finally {
            setGenerating(false);
        }
    };

    const handleDeleteClick = (policy: Policy) => {
        setPolicyToDelete(policy);
        setDeleteDialogOpen(true);
    };

    const handleConfirmDelete = async () => {
        if (!policyToDelete) return;
        setGenerating(true);
        try {
            await api.deletePolicy(policyToDelete.id);
            alert('Policy deleted successfully!');
            setDeleteDialogOpen(false);
            fetchPolicies();
        } catch (error) {
            console.error('Delete failed:', error);
            alert('Failed to delete policy.');
        } finally {
            setGenerating(false);
        }
    };

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!selectedFile) return;
        setUploading(true);
        try {
            const response = await api.uploadPolicies(selectedFile);
            alert(response.data.message || 'Policies uploaded successfully!');
            setSelectedFile(null);
            fetchPolicies();
            setActiveTab(3);
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Failed to upload policies. Please check the file format.');
        } finally {
            setUploading(false);
        }
    };

    return (
        <Box sx={{
            // mt: -2,
            mx: -2,
            mb: -2,
            width: 'calc(100% + 32px)',
            bgcolor: '#fbfbfb',
            minHeight: '100%',
            // overflowX: 'hidden',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center'
        }}>
            {/* Header / Tabs Section */}
            <Box sx={{
                borderBottom: '1px solid #E8E8E8',
                px: 4,
                mt: -2,
                bgcolor: '#fff',
                width: '100%',
                display: 'flex',
                justifyContent: 'center'
            }}>
                <Tabs
                    value={activeTab}
                    onChange={(_, val) => setActiveTab(val)}
                    variant="scrollable"
                    scrollButtons="auto"
                    sx={{
                        width: '100%',
                        maxWidth: '1440px',
                        '& .MuiTabs-indicator': {
                            backgroundColor: '#6366f1',
                            height: 2,
                        },
                        '& .MuiTab-root': {
                            textTransform: 'none',
                            fontWeight: 500,
                            fontSize: '0.85rem',
                            color: '#666D73',
                            minWidth: 'auto',
                            px: 3,
                            py: 1.5,
                            border: 'none',
                            outline: 'none',
                            boxShadow: 'none',
                            '&:focus': {
                                outline: 'none',
                                border: 'none',
                                boxShadow: 'none',
                            },
                            '&:focus-visible': {
                                outline: 'none',
                                border: 'none',
                                boxShadow: 'none',
                            },
                            '&.Mui-selected': {
                                color: '#6366f1',
                                boxShadow: 'none',
                                border: 'none',
                                outline: 'none',
                            },
                        }
                    }}
                >
                    <Tab label="Generate from Regulation" disableRipple />
                    <Tab label="Add Policy Manually" disableRipple />
                    <Tab label="Upload from Excel" disableRipple />
                    <Tab label="Existing Policies" disableRipple />
                </Tabs>
            </Box>

            <Box sx={{ mt: 6, display: 'flex', justifyContent: 'center', width: '100%' }}>
                {activeTab === 0 && (
                    <Box sx={{ width: '100%', maxWidth: '1440px', px: 4 }}>
                        <Paper
                            elevation={0}
                            sx={{
                                border: '1px solid #E8E8E8',
                                borderRadius: '12px',
                                overflow: 'hidden',
                                boxShadow: 'none'
                            }}
                        >
                            {/* Paper Header */}
                            <Box sx={{ p: 2, borderBottom: '1px solid #E8E8E8' }}>
                                <Typography sx={{ fontSize: '1.25rem', fontWeight: 600, color: '#131313', mb: 0.5 }}>
                                    Generate Policy from Regulatory Text
                                </Typography>
                                <Typography sx={{ fontSize: '0.85rem', color: '#666D73' }}>
                                    Enter regulatory text to automatically generate compliance policies using AI.
                                </Typography>
                            </Box>

                            {/* Paper Body */}
                            <Box sx={{ p: 3 }}>
                                <Box sx={{ mb: 3 }}>
                                    <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                        Regulation Name<span style={{ color: 'red' }}>*</span>
                                    </Typography>
                                    <TextField
                                        fullWidth
                                        variant="outlined"
                                        size="small"
                                        value={regulationName}
                                        onChange={(e) => setRegulationName(e.target.value)}
                                        sx={{
                                            '& .MuiOutlinedInput-root': {
                                                borderRadius: 2,
                                                fontSize: '0.9rem',
                                                '& fieldset': { borderColor: '#E8E8E8' },
                                                '&:hover fieldset': { borderColor: '#6366f1' },
                                                '&.Mui-focused fieldset': { borderColor: '#6366f1', borderWidth: '1px' },
                                            }
                                        }}
                                    />
                                </Box>

                                <Box sx={{ mb: 3 }}>
                                    <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                        Regulatory Text<span style={{ color: 'red' }}>*</span>
                                    </Typography>
                                    <TextField
                                        fullWidth
                                        multiline
                                        rows={8}
                                        variant="outlined"
                                        value={regulatoryText}
                                        onChange={(e) => setRegulatoryText(e.target.value)}
                                        sx={{
                                            '& .MuiOutlinedInput-root': {
                                                borderRadius: 2,
                                                fontSize: '0.9rem',
                                                '& fieldset': { borderColor: '#E8E8E8' },
                                                '&:hover fieldset': { borderColor: '#6366f1' },
                                                '&.Mui-focused fieldset': { borderColor: '#6366f1', borderWidth: '1px' },
                                            }
                                        }}
                                    />
                                </Box>

                                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
                                    <Button
                                        variant="contained"
                                        disabled={generating || !regulatoryText}
                                        onClick={handleGenerate}
                                        sx={{
                                            textTransform: 'none',
                                            bgcolor: '#6366f1',
                                            color: '#fff',
                                            borderRadius: '6px',
                                            boxShadow: 'none',
                                            fontWeight: 500,
                                            px: 3.5,
                                            py: 0.8,
                                            fontSize: '0.85rem',
                                            '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' },
                                            '&:disabled': { bgcolor: '#E2E8F0', color: '#94A3B8' }
                                        }}
                                    >
                                        {generating ? <CircularProgress size={20} color="inherit" /> : 'Save & Next'}
                                    </Button>
                                </Box>
                            </Box>
                        </Paper>
                    </Box>
                )}

                {activeTab === 1 && (
                    <Box sx={{ width: '100%', maxWidth: '1440px', px: 4 }}>
                        <Paper
                            elevation={0}
                            sx={{
                                border: '1px solid #E8E8E8',
                                borderRadius: '12px',
                                overflow: 'hidden',
                                boxShadow: 'none',
                                bgcolor: '#fff'
                            }}
                        >
                            {/* Paper Header */}
                            <Box sx={{ p: 2, px: 3, borderBottom: '1px solid #E8E8E8' }}>
                                <Typography sx={{ fontSize: '1.25rem', fontWeight: 600, color: '#131313', mb: 0.5 }}>
                                    Add Policy Manually
                                </Typography>
                                <Typography sx={{ fontSize: '0.85rem', color: '#666D73' }}>
                                    Create a custom compliance policy with your own rules.
                                </Typography>
                            </Box>

                            {/* Paper Body */}
                            <Box sx={{ p: 3, px: 4 }}>
                                <Grid container spacing={2}>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Policy ID<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            variant="outlined"
                                            size="small"
                                            placeholder="Enter Display Name"
                                            value={manualPolicy.policyId}
                                            onChange={(e) => handleManualChange('policyId', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        />
                                    </Grid>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Policy Version<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            variant="outlined"
                                            size="small"
                                            placeholder="Enter Display Name"
                                            value={manualPolicy.version}
                                            onChange={(e) => handleManualChange('version', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Policy Name<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            variant="outlined"
                                            size="small"
                                            placeholder="Enter Display Name"
                                            value={manualPolicy.name}
                                            onChange={(e) => handleManualChange('name', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        >
                                        </TextField>
                                    </Grid>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Severity<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            select
                                            fullWidth
                                            variant="outlined"
                                            size="small"
                                            value={manualPolicy.severity}
                                            onChange={(e) => handleManualChange('severity', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        >
                                            <MenuItem value="" disabled>Select Severity</MenuItem>
                                            <MenuItem value="critcal">Critcal</MenuItem>
                                            <MenuItem value="high">High</MenuItem>
                                            <MenuItem value="medium">Medium</MenuItem>
                                            <MenuItem value="low">Low</MenuItem>
                                        </TextField>
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Category<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            select
                                            fullWidth
                                            variant="outlined"
                                            size="small"
                                            value={manualPolicy.category}
                                            onChange={(e) => handleManualChange('category', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        >
                                            <MenuItem value="" disabled>Select Category</MenuItem>
                                            {["Illegal Activities", "Violence", "Financial Fraud", "Self Harm", "Extremism", "Medical", "Legal", "Hate Speech", "Sexual Content", "Privacy", "Hallucination", "Overconfidence",
                                                "fairness", "privacy", "reliability", "security", "safety", "governance", "transparency", 
                                            ]
                                                .map((c) => (
                                                    <MenuItem key={c} value={c}>
                                                        {c}
                                                    </MenuItem>
                                                ))}
                                        </TextField>
                                    </Grid>
                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Package
                                        </Typography>
                                        <TextField
                                            select
                                            fullWidth
                                            variant="outlined"
                                            size="small"
                                            value={manualPolicy.packageId}
                                            onChange={(e) => handleManualChange('packageId', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        >
                                            <MenuItem value="">Select Package</MenuItem>
                                            {packages.map((pkg: {id: string, name: string, description: string}) => (
                                                <MenuItem key={pkg.id} value={pkg.id}>
                                                    {pkg.name}
                                                </MenuItem>
                                            ))}
                                        </TextField>
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }} sx={{ display: 'flex', alignItems: 'center', mt: 3 }}>
                                        <FormControlLabel
                                            control={
                                                <Checkbox
                                                    checked={manualPolicy.active}
                                                    onChange={(e) => setManualPolicy(prev => ({ ...prev, active: e.target.checked }))}
                                                    sx={{
                                                        color: '#E8E8E8',
                                                        '&.Mui-checked': {
                                                            color: '#6366f1',
                                                        },
                                                    }}
                                                />
                                            }
                                            label={
                                                <Typography sx={{ fontSize: '0.85rem', fontWeight: 500, color: '#313131' }}>
                                                    Active
                                                </Typography>
                                            }
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 12 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Description<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            multiline
                                            rows={4}
                                            variant="outlined"
                                            placeholder="Write Description"
                                            value={manualPolicy.description}
                                            onChange={(e) => handleManualChange('description', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        />
                                    </Grid>
                                </Grid>

                                <Box sx={{ mt: 4, mb: 2 }}>
                                    <Typography sx={{ fontSize: '1.1rem', fontWeight: 600, color: '#131313' }}>
                                        Rules
                                    </Typography>
                                    <Typography sx={{ fontSize: '0.8rem', color: '#666D73', mb: 2 }}>
                                        Add compliance rules (one per line). Format: rule text | severity where severity is critical, high, medium, or low.
                                    </Typography>

                                    <Box sx={{ mb: 3 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Rules (one per line)<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            multiline
                                            rows={6}
                                            variant="outlined"
                                            placeholder="Write Rules (one per line)"
                                            value={manualPolicy.rules}
                                            onChange={(e) => handleManualChange('rules', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        />
                                    </Box>

                                    <Box sx={{ mb: 3 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Regulatory References (comma-separated)
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            variant="outlined"
                                            size="small"
                                            placeholder="Enter Custom Endpoint URL"
                                            value={manualPolicy.references}
                                            onChange={(e) => handleManualChange('references', e.target.value)}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    borderRadius: 2,
                                                    fontSize: '0.9rem',
                                                    '& fieldset': { borderColor: '#E8E8E8' },
                                                }
                                            }}
                                        />
                                    </Box>
                                </Box>

                                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
                                    <Button
                                        variant="contained"
                                        disabled={generating || !manualPolicy.name || !manualPolicy.policyId}
                                        onClick={handleCreateManual}
                                        sx={{
                                            textTransform: 'none',
                                            bgcolor: '#6366f1',
                                            color: '#fff',
                                            borderRadius: '6px',
                                            boxShadow: 'none',
                                            fontWeight: 500,
                                            px: 4,
                                            py: 1,
                                            fontSize: '0.85rem',
                                            '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' },
                                            // '&:disabled': { bgcolor: '#E2E8F0', color: '#94A3B8' }
                                        }}
                                    >
                                        {generating ? <CircularProgress size={20} color="inherit" /> : 'Create Policy'}
                                    </Button>
                                </Box>
                            </Box>
                        </Paper>
                    </Box>
                )}

                {activeTab === 2 && (
                    <Box sx={{ width: '100%', maxWidth: '1440px', px: 4 }}>
                        <Paper
                            elevation={0}
                            sx={{
                                border: '1px solid #E8E8E8',
                                borderRadius: '12px',
                                overflow: 'hidden',
                                boxShadow: 'none',
                                bgcolor: '#fff'
                            }}
                        >
                            {/* Paper Header */}
                            <Box sx={{ p: 2, px: 3, borderBottom: '1px solid #E8E8E8' }}>
                                <Typography sx={{ fontSize: '1.25rem', fontWeight: 600, color: '#131313', mb: 0.5 }}>
                                    Upload Policies from Excel
                                </Typography>
                                <Typography sx={{ fontSize: '0.85rem', color: '#666D73' }}>
                                    Upload an Excel file (.xlsx or .xls) containing policy data.
                                </Typography>
                            </Box>

                            {/* Paper Body */}
                            <Box sx={{ p: 4, px: 4 }}>
                                <Typography sx={{ fontSize: '0.9rem', fontWeight: 600, mb: 2, color: '#131313' }}>
                                    Excel Format - Two Options:
                                </Typography>

                                <Box sx={{ bgcolor: '#F8F9FA', p: 2.5, borderRadius: 2, mb: 2, border: '1px solid #E8E8E8' }}>
                                    <Typography sx={{ fontSize: '0.85rem', fontWeight: 600, mb: 1, color: '#131313' }}>
                                        Option 1 - Simple Format (Descriptions Only):
                                    </Typography>
                                    <Typography sx={{ fontSize: '0.8rem', color: '#666D73', lineHeight: 1.6 }}>
                                        Just one column with policy descriptions (one per row)<br />
                                        System will auto-generate policy IDs, names, and extract rules
                                    </Typography>
                                </Box>

                                <Box sx={{ bgcolor: '#F8F9FA', p: 2.5, borderRadius: 2, mb: 4, border: '1px solid #E8E8E8' }}>
                                    <Typography sx={{ fontSize: '0.85rem', fontWeight: 600, mb: 1, color: '#131313' }}>
                                        Option 2 - Full Format (All Fields):
                                    </Typography>
                                    <Typography sx={{ fontSize: '0.8rem', color: '#666D73', lineHeight: 1.6 }}>
                                        Column 1: policy_id - Unique identifier for the policy<br />
                                        Column 2: policy_name - Name of the policy<br />
                                        Column 3: description - Policy description<br />
                                        Column 4: category - Category (safety, fairness, privacy, transparency, etc.)<br />
                                        Column 5: version - Version number (e.g., 1.0.0)<br />
                                        Column 6: active - TRUE or FALSE<br />
                                        Column 7: rules - Rules, separated by semicolons (;)<br />
                                        Column 8: rule_severities - Severities for each rule, separated by semicolons (;)
                                    </Typography>
                                </Box>

                                <Typography sx={{ fontSize: '0.9rem', fontWeight: 600, mb: 2, color: '#131313' }}>
                                    Choose an Excel file
                                </Typography>

                                <Box
                                    component="label"
                                    sx={{
                                        border: '1px dashed #E8E8E8',
                                        borderRadius: '8px',
                                        p: 3,
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        bgcolor: selectedFile ? alpha('#6366f1', 0.05) : '#fff',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s',
                                        borderColor: selectedFile ? '#6366f1' : '#E8E8E8',
                                        '&:hover': {
                                            borderColor: '#6366f1',
                                            bgcolor: alpha('#6366f1', 0.01)
                                        }
                                    }}
                                >
                                    <input
                                        type="file"
                                        hidden
                                        accept=".csv,.xlsx,.xls"
                                        onChange={handleFileChange}
                                    />
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                        <Box sx={{
                                            bgcolor: alpha('#6366f1', 0.05),
                                            p: 1.2,
                                            borderRadius: 2,
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center'
                                        }}>
                                            <CloudUploadOutlined sx={{ fontSize: 24, color: '#6366f1' }} />
                                        </Box>
                                        <Box>
                                            <Typography sx={{ fontSize: '0.85rem', fontWeight: 500, color: '#313131' }}>
                                                {selectedFile ? selectedFile.name : <><span style={{ color: '#6366f1', cursor: 'pointer' }}>Click to Upload</span> or drag and drop</>}
                                            </Typography>
                                            <Typography sx={{ fontSize: '0.75rem', color: '#94A3B8' }}>
                                                {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : '(Max. File size: 25 MB)'}
                                            </Typography>
                                        </Box>
                                    </Box>
                                </Box>

                                <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
                                    <Button
                                        variant="contained"
                                        disabled={uploading || !selectedFile}
                                        onClick={handleUpload}
                                        sx={{
                                            textTransform: 'none',
                                            bgcolor: '#6366f1',
                                            color: '#fff',
                                            borderRadius: '6px',
                                            boxShadow: 'none',
                                            fontWeight: 500,
                                            px: 4,
                                            py: 1,
                                            fontSize: '0.85rem',
                                            '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' },
                                            '&:disabled': { bgcolor: '#E2E8F0', color: '#94A3B8' }
                                        }}
                                    >
                                        {uploading ? <CircularProgress size={20} color="inherit" /> : 'Import Policies'}
                                    </Button>
                                </Box>

                                {/* <Box sx={{ mt: 4, display: 'flex', justifyContent: 'flex-end' }}>
                                    <Button
                                        variant="contained"
                                        disabled={generating}
                                        sx={{
                                            textTransform: 'none',
                                            bgcolor: '#6366f1',
                                            color: '#fff',
                                            borderRadius: '6px',
                                            boxShadow: 'none',
                                            fontWeight: 500,
                                            px: 4,
                                            py: 1,
                                            fontSize: '0.85rem',
                                            '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' },
                                            // '&:disabled': { bgcolor: '#E2E8F0', color: '#94A3B8' }
                                        }}
                                    >
                                        {generating ? <CircularProgress size={20} color="inherit" /> : 'Import Policies'}
                                    </Button>
                                </Box> */}
                            </Box>
                        </Paper>
                    </Box>
                )}

                {activeTab === 3 && (
                    <Box sx={{ width: '100%', maxWidth: '1440px', px: 4 }}>
                        <Paper
                            elevation={0}
                            sx={{
                                border: '1px solid #E8E8E8',
                                borderRadius: '12px',
                                overflow: 'hidden',
                                bgcolor: '#fff'
                            }}
                        >
                            {/* Paper Header */}
                            <Box sx={{ p: 2, px: 3, borderBottom: '1px solid #E8E8E8', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Box>
                                    <Typography sx={{ fontSize: '1.25rem', fontWeight: 600, color: '#131313', mb: 0.5 }}>
                                        Existing Policies
                                    </Typography>
                                    <Typography sx={{ fontSize: '0.85rem', color: '#666D73' }}>
                                        Manage and view all your current compliance and safety policies.
                                    </Typography>
                                </Box>
                                <TextField
                                    size="small"
                                    placeholder="Search policies..."
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <SearchOutlined sx={{ fontSize: 20, color: '#94A3B8' }} />
                                            </InputAdornment>
                                        ),
                                    }}
                                    sx={{
                                        width: 250,
                                        '& .MuiOutlinedInput-root': {
                                            borderRadius: 2,
                                            fontSize: '0.85rem',
                                            '& fieldset': { borderColor: '#E8E8E8' },
                                        }
                                    }}
                                />
                                <IconButtonSmall
                                    onClick={fetchPolicies}
                                    sx={{ ml: 1, color: '#6366f1', borderColor: alpha('#6366f1', 0.1) }}
                                >
                                    <RefreshOutlined sx={{ fontSize: 18 }} />
                                </IconButtonSmall>
                                <Button
                                    variant="outlined"
                                    onClick={fetchPolicies}
                                    startIcon={<RefreshOutlined />}
                                    size="small"
                                    sx={{
                                        ml: 2,
                                        textTransform: 'none',
                                        borderRadius: 2,
                                        borderColor: '#E8E8E8',
                                        color: '#666D73',
                                        display: { xs: 'none', md: 'flex' }
                                    }}
                                >
                                    Refresh List
                                </Button>
                            </Box>

                            {/* Paper Body */}
                            <Box sx={{ p: 0 }}>
                                {loading ? (
                                    <Box sx={{ textAlign: 'center', py: 8 }}><CircularProgress /></Box>
                                ) : (
                                    <Box>
                                        <Box sx={{
                                            display: 'grid',
                                            gridTemplateColumns: '2fr 1fr 1fr 100px',
                                            px: 3,
                                            py: 1.5,
                                            bgcolor: '#fbfbfb',
                                            borderBottom: '1px solid #E8E8E8'
                                        }}>
                                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#666D73', textTransform: 'uppercase' }}>Policy Name</Typography>
                                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#666D73', textTransform: 'uppercase' }}>Category</Typography>
                                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#666D73', textTransform: 'uppercase' }}>Status</Typography>
                                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#666D73', textTransform: 'uppercase', textAlign: 'right' }}>Actions</Typography>
                                        </Box>

                                        {policies.map((p, i) => (
                                            <Box key={i} sx={{
                                                display: 'grid',
                                                gridTemplateColumns: '2fr 1fr 1fr 100px',
                                                px: 3,
                                                py: 2,
                                                borderBottom: i === policies.length - 1 ? 'none' : '1px solid #f1f5f9',
                                                alignItems: 'center',
                                                '&:hover': { bgcolor: '#fcfcfc' }
                                            }}>
                                                <Typography sx={{ fontWeight: 500, fontSize: '0.9rem', color: '#131313' }}>{p.name}</Typography>
                                                <Typography sx={{ fontSize: '0.85rem', color: '#666D73' }}>{p.category}</Typography>
                                                <Box>
                                                    <Box sx={{
                                                        display: 'inline-block',
                                                        px: 1.5,
                                                        py: 0.5,
                                                        borderRadius: '12px',
                                                        fontSize: '0.7rem',
                                                        fontWeight: 500,
                                                        bgcolor: p.active === true ? alpha('#10b981', 0.1) : alpha('#f59e0b', 0.1),
                                                        color: p.active === true ? '#059669' : '#d97706'
                                                    }}>
                                                        {p.active === true ? 'Active' : 'Unknown'}
                                                    </Box>
                                                </Box>
                                                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
                                                    <IconButtonSmall onClick={() => handleEditClick(p)}><EditOutlined sx={{ fontSize: 18 }} /></IconButtonSmall>
                                                    <IconButtonSmall onClick={() => handleDeleteClick(p)} sx={{ color: '#ef4444' }}><DeleteOutline sx={{ fontSize: 18 }} /></IconButtonSmall>
                                                </Box>
                                            </Box>
                                        ))}

                                        {policies.length === 0 && (
                                            <Box sx={{ textAlign: 'center', py: 8 }}>
                                                <Typography color="text.secondary">No policies found.</Typography>
                                            </Box>
                                        )}
                                    </Box>
                                )}
                            </Box>
                        </Paper>
                    </Box>
                )}
            </Box>

            {/* Edit Policy Dialog */}
            <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)} maxWidth="sm" fullWidth>
                <DialogTitle sx={{ borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography sx={{ fontWeight: 600 }}>Edit Policy</Typography>
                    <IconButton onClick={() => setEditDialogOpen(false)} size="small"><CloseOutlined /></IconButton>
                </DialogTitle>
                <DialogContent sx={{ p: 3 }}>
                    <Grid container spacing={2} sx={{ mt: 0.5 }}>
                        <Grid size={{ xs: 12 }}>
                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>Policy Name</Typography>
                            <TextField
                                fullWidth size="small" value={editForm.name}
                                onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                            />
                        </Grid>
                        <Grid size={{ xs: 6 }}>
                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>Category</Typography>
                            <TextField
                                select fullWidth size="small" value={editForm.category}
                                onChange={(e) => setEditForm({ ...editForm, category: e.target.value })}
                            >
                                {["Illegal Activities", "Violence", "Financial Fraud", "Self Harm", "Extremism", "Medical", "Legal", "Hate Speech", "Sexual Content", "Privacy", "Hallucination", "Overconfidence",
                                    "fairness", "privacy", "reliability", "security", "safety", "governance",
                                ]
                                    .map((c) => <MenuItem key={c} value={c}>{c}</MenuItem>)}
                            </TextField>
                        </Grid>
                        <Grid size={{ xs: 6 }}>
                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>Severity</Typography>
                            <TextField
                                select fullWidth size="small" value={editForm.severity}
                                onChange={(e) => setEditForm({ ...editForm, severity: e.target.value })}
                            >
                                <MenuItem value="critical">Critical</MenuItem>
                                <MenuItem value="high">High</MenuItem>
                                <MenuItem value="medium">Medium</MenuItem>
                                <MenuItem value="low">Low</MenuItem>
                            </TextField>
                        </Grid>
                        <Grid size={{ xs: 6 }}>
                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>Package</Typography>
                            <TextField
                                select fullWidth size="small" value={editForm.packageId}
                                onChange={(e) => setEditForm({ ...editForm, packageId: e.target.value })}
                            >
                                <MenuItem value="">Select Package</MenuItem>
                               {packages.map((pkg: {id: string, name: string, description: string}) => (
                                    <MenuItem key={pkg.id} value={pkg.id}>{pkg.name}</MenuItem>
                                ))}
                            </TextField>
                        </Grid>
                        <Grid size={{ xs: 12 }}>
                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>Version</Typography>
                            <TextField
                                fullWidth size="small" value={editForm.version}
                                onChange={(e) => setEditForm({ ...editForm, version: e.target.value })}
                            />
                        </Grid>
                        <Grid size={{ xs: 12 }}>
                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>Description</Typography>
                            <TextField
                                fullWidth multiline rows={3} size="small" value={editForm.description}
                                onChange={(e) => setEditForm({ ...editForm, description: e.target.value })}
                            />
                        </Grid>
                        <Grid size={{ xs: 12 }}>
                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 0.5 }}>Rules (one per line: text | severity)</Typography>
                            <TextField
                                fullWidth multiline rows={4} size="small" value={editForm.rules}
                                onChange={(e) => setEditForm({ ...editForm, rules: e.target.value })}
                            />
                        </Grid>
                        <Grid size={{ xs: 12 }}>
                            <FormControlLabel
                                control={<Checkbox checked={editForm.active} onChange={(e) => setEditForm({ ...editForm, active: e.target.checked })} />}
                                label={<Typography sx={{ fontSize: '0.85rem' }}>Active</Typography>}
                            />
                        </Grid>
                    </Grid>
                </DialogContent>
                <DialogActions sx={{ p: 2, borderTop: '1px solid #f1f5f9' }}>
                    <Button onClick={() => setEditDialogOpen(false)} sx={{ textTransform: 'none', color: '#64748b' }}>Cancel</Button>
                    <Button
                        variant="contained"
                        onClick={handleUpdatePolicy}
                        disabled={generating}
                        startIcon={generating ? <CircularProgress size={16} color="inherit" /> : <SaveOutlined />}
                        sx={{ textTransform: 'none', bgcolor: '#6366f1', px: 3 }}
                    >
                        Save Changes
                    </Button>
                </DialogActions>
            </Dialog>

            {/* Delete Confirmation Dialog */}
            <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
                <DialogTitle sx={{ fontWeight: 600 }}>Confirm Delete</DialogTitle>
                <DialogContent>
                    <Typography sx={{ fontSize: '0.9rem', color: '#64748b' }}>
                        Are you sure you want to delete the policy "<strong>{policyToDelete?.name}</strong>"? This action cannot be undone.
                    </Typography>
                </DialogContent>
                <DialogActions sx={{ p: 2 }}>
                    <Button onClick={() => setDeleteDialogOpen(false)} sx={{ textTransform: 'none', color: '#64748b' }}>Cancel</Button>
                    <Button
                        variant="contained"
                        onClick={handleConfirmDelete}
                        disabled={generating}
                        sx={{ textTransform: 'none', bgcolor: '#ef4444', '&:hover': { bgcolor: '#dc2626' } }}
                    >
                        {generating ? <CircularProgress size={20} color="inherit" /> : 'Delete Policy'}
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default Policy;
