import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Paper,
    Tabs,
    Tab,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Grid,
    TextField,
    MenuItem,
    Button,
    CircularProgress,
    Slider,
    InputAdornment
} from '@mui/material';
import {
    Visibility as EyeIcon,
    FilterList as FilterIcon,
    Close as CloseIcon,
    CategoryOutlined,
    WorkOutline,
    WbSunnyOutlined,
    ViewHeadlineOutlined,
    HourglassEmptyOutlined,
    BoltOutlined,
    CheckCircle as CheckCircleIcon,
    Search as SearchIcon
} from '@mui/icons-material';
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    IconButton,
    alpha,
    Card,
    CardContent
} from '@mui/material';
import {
    PieChart,
    Pie,
    Tooltip,
    ResponsiveContainer,
    Legend,
    Cell
} from 'recharts';
import { api } from '../api/client';
import type { AIModel } from '../Types';

const COLORS = ['#6366F1', '#8B5CF6', '#EC4899', '#F43F5E', '#F97316', '#EAB308', '#22C55E', '#06B6D4'];

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

const InfoCard: React.FC<{ icon: React.ReactNode, label: string, value: string | number }> = ({ icon, label, value }) => (
    <Card elevation={0} sx={{ border: '1px solid #F1F5F9', borderRadius: 2, bgcolor: '#fff' }}>
        <CardContent sx={{ display: 'flex', alignItems: 'center', gap: 2, p: '16px !important' }}>
            <Box sx={{
                bgcolor: alpha('#6366F1', 0.08),
                p: 1.2,
                borderRadius: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
            }}>
                {icon}
            </Box>
            <Box>
                <Typography sx={{ fontSize: '0.75rem', fontWeight: 500, color: '#666D73', mb: 0.2 }}>{label}</Typography>
                <Typography sx={{ fontSize: '1rem', fontWeight: 600, color: '#131313' }}>{value}</Typography>
            </Box>
        </CardContent>
    </Card>
);

function CustomTabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`model-tabpanel-${index}`}
            aria-labelledby={`model-tab-${index}`}
            {...other}
        >
            {value === index && (
                <Box sx={{ pt: 0 }}>
                    {children}
                </Box>
            )}
        </div>
    );
}

const Model: React.FC = () => {
    const [tabValue, setTabValue] = useState(0);
    const [models, setModels] = useState<AIModel[]>([]);
    const [loading, setLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [formData, setFormData] = useState({
        name: '',
        provider: 'openai',
        model_id: '',
        api_key: '',
        description: '',
        model_type: 'api',
        custom_endpoint: '',
        max_tokens: '4096',
        tags: '',
        temperature: 0.7
    });

    const [filters, setFilters] = useState({
        search: '',
        type: 'all',
        provider: 'all',
        status: 'all'
    });

    const [detailsOpen, setDetailsOpen] = useState(false);
    const [selectedModel, setSelectedModel] = useState<AIModel | null>(null);

    const handleViewMore = (model: AIModel) => {
        setSelectedModel(model);
        setDetailsOpen(true);
    };

    const fetchModels = async () => {
        setLoading(true);
        try {
            const response = await api.getModels();
            console.log('response.data.models', response.data.models)
            setModels(response.data.models);
        } catch (error) {
            console.error('Error fetching models:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchModels();
    }, []);

    const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

    const handleRegister = async () => {
        setSubmitting(true);
        try {
            await api.registerModel({
                name: formData.name,
                provider: formData.provider,
                id: formData.model_id,
                model_type: formData.model_type,
                description: formData.description,
                max_tokens: parseInt(formData.max_tokens),
                temperature: formData.temperature,
                tags: formData.tags ? formData.tags.split(',').map(t => t.trim()) : [],
                config: {
                    api_key: formData.api_key,
                    endpoint_url: formData.custom_endpoint
                }
            });
            alert('Model registered successfully!');
            setFormData({
                name: '',
                provider: 'openai',
                model_id: '',
                api_key: '',
                description: '',
                model_type: 'api',
                custom_endpoint: '',
                max_tokens: '4096',
                tags: '',
                temperature: 0.7
            });
            setTabValue(0);
            fetchModels();
        } catch (error) {
            console.error('Registration failed:', error);
            alert('Failed to register model.');
        } finally {
            setSubmitting(false);
        }
    };

    // Mock model data with descriptions
    const modelDescriptions: { [key: string]: string } = {
        'GPT-4': 'Utilizes the advanced Gemini Pro model; API key required',
        'Grok AI': 'Leverages the powerful Grok 1.5 model; API key required',
        'Bard': 'Powered by the cutting-edge Minerva model; API key required',
        'Claude': 'Employs the sophisticated Luminous model; API key required',
        'Llama 2': 'Based on the robust Titan model; API key required',
        'Falcon': 'Runs on the efficient Claude 1 model; API key needed',
        'Bloom': 'Built on the versatile Falcon LLM; API key required',
        'LaMDA': 'Integrated with the innovative Bloom model; API key required'
    };

    const filteredModels = models.filter(model => {
        const matchesSearch = model.name.toLowerCase().includes(filters.search.toLowerCase()) ||
            (model.provider || '').toLowerCase().includes(filters.search.toLowerCase());
        const matchesProvider = filters.provider === 'all' || model.provider?.toLowerCase() === filters.provider.toLowerCase();
        const matchesType = filters.type === 'all' || model.model_type?.toLowerCase() === filters.type.toLowerCase();
        const matchesStatus = filters.status === 'all' || model.status === filters.status;
        return matchesSearch && matchesType && matchesProvider && matchesStatus;
    });

    return (
        <Box sx={{ mt: -2, mx: -2, px:0,  width: '100%' }}>

            <Box sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                borderBottom: 1,
                borderColor: '#E8E8E8',
                backgroundColor: '#FFFFFF',
                pr: 2
            }}>
                <Tabs
                    value={tabValue}
                    onChange={handleTabChange}
                    aria-label="model management tabs"
                    sx={{
                        '& .MuiTab-root': {
                            textTransform: 'none',
                            fontWeight: 500,
                            fontSize: '0.875rem',
                            color: '#666D73',
                            minHeight: 48,
                            '&.Mui-selected': {
                                color: '#6366F1',
                                fontWeight: 600
                            }
                        },
                        '& .MuiTabs-indicator': {
                            backgroundColor: '#6366F1',
                            height: 3
                        }
                    }}
                >
                    <Tab label="Registered Models" />
                    <Tab label="Register New Model" />
                    <Tab label="Model Stats" />
                </Tabs>
            </Box>
            <Paper sx={{ width: '100%', height: '100%', boxShadow: '0 1px 3px rgba(0,0,0,0.05)', borderRadius: 2, mt: 2 }}>
                <CustomTabPanel value={tabValue} index={0}>
                    <Box sx={{ p: 2 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', mb: 3 }}>
                            <Box>
                                <Typography variant="h6" sx={{ fontWeight: 600, mb: 0.5, color: '#131313' }}>
                                    Models Name
                                </Typography>
                                <Typography variant="body2" sx={{ fontWeight: 400, color: '#666D73', fontSize: '14px' }}>
                                    {filteredModels.filter(m => m.status === 'inactive').length} Models are Inactive
                                </Typography>
                            </Box>

                            <Box sx={{ display: 'flex', gap: 2 }}>
                                <TextField
                                    size="small"
                                    placeholder="Search models..."
                                    value={filters.search}
                                    onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                                    InputProps={{
                                        startAdornment: (
                                            <InputAdornment position="start">
                                                <SearchIcon sx={{ color: '#94A3B8', fontSize: 20 }} />
                                            </InputAdornment>
                                        ),
                                    }}
                                    sx={{ width: 240, '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#fff' } }}
                                />
                                <TextField
                                    select
                                    size="small"
                                    value={filters.type}
                                    onChange={(e) => setFilters({ ...filters, type: e.target.value })}
                                    sx={{ width: 140, '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#fff' } }}
                                >
                                    <MenuItem value="all">All Types</MenuItem>
                                    <MenuItem value="api">API</MenuItem>
                                    <MenuItem value="ollama">Ollama</MenuItem>
                                    <MenuItem value="huggingface">HuggingFace</MenuItem>
                                    <MenuItem value="uploaded">Uploaded</MenuItem>
                                </TextField>
                                <TextField
                                    select
                                    size="small"
                                    value={filters.provider}
                                    onChange={(e) => setFilters({ ...filters, provider: e.target.value })}
                                    sx={{ width: 140, '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#fff' } }}
                                >
                                    <MenuItem value="all">All Providers</MenuItem>
                                    <MenuItem value="openai">OpenAI</MenuItem>
                                    <MenuItem value="anthropic">Anthropic</MenuItem>
                                    <MenuItem value="ollama">Ollama</MenuItem>
                                    <MenuItem value="huggingface">HuggingFace</MenuItem>
                                </TextField>
                                <TextField
                                    select
                                    size="small"
                                    value={filters.status}
                                    onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                                    sx={{ width: 140, '& .MuiOutlinedInput-root': { borderRadius: 2, bgcolor: '#fff' } }}
                                >
                                    <MenuItem value="all">All Status</MenuItem>
                                    <MenuItem value="active">Active</MenuItem>
                                    <MenuItem value="inactive">Inactive</MenuItem>
                                </TextField>
                            </Box>
                        </Box>
                    </Box>

                    {loading ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
                            <CircularProgress />
                        </Box>
                    ) : (
                        <TableContainer sx={{ borderRadius: '0 0 8px 8px' }}>
                            <Table>
                                <TableHead>
                                    <TableRow sx={{ '& th': { borderBottom: 'none', color: '#131313', fontWeight: 500, py: 1, backgroundColor: '#F3F3F3' } }}>
                                        <TableCell>Model Name</TableCell>
                                        <TableCell>Status</TableCell>
                                        <TableCell>Description</TableCell>
                                        <TableCell>Provider</TableCell>
                                        <TableCell>Type</TableCell>
                                        <TableCell align="right">Max Tokens</TableCell>
                                        <TableCell align="center">Actions</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    {filteredModels.map((model) => (
                                        <TableRow
                                            key={model.id}
                                            hover
                                            sx={{
                                                '&:last-child td, &:last-child th': { border: 0 },
                                                '& td': { py: 1 }
                                            }}
                                        >
                                            <TableCell sx={{ fontWeight: 500, color: '#666D73', fontSize: '14px' }}>
                                                {model.name}
                                            </TableCell>
                                            <TableCell>
                                                <Chip
                                                    label={model.status === 'active' ? 'Active' : 'Inactive'}
                                                    size="small"
                                                    sx={{
                                                        fontWeight: 500,
                                                        fontSize: '0.75rem',
                                                        bgcolor: model.status === 'active' ? '#DCFCE7' : '#FEE2E2',
                                                        color: model.status === 'active' ? '#166534' : '#991B1B',
                                                        borderRadius: '6px',
                                                        height: '24px'
                                                    }}
                                                />
                                            </TableCell>
                                            <TableCell sx={{ color: '#666D73', fontSize: '14px', maxWidth: 300 }}>
                                                {modelDescriptions[model.name] || 'AI model; API key required'}
                                            </TableCell>
                                            <TableCell sx={{ color: '#666D73', fontSize: '14px' }}>
                                                {model.provider || 'OpenAI'}
                                            </TableCell>
                                            <TableCell sx={{ color: '#666D73', fontSize: '14px' }}>
                                                {model.model_type || 'API'}
                                            </TableCell>
                                            <TableCell align="right" sx={{ fontWeight: 500, color: '#666D73' }}>
                                                {model.max_tokens?.toLocaleString() || '4,096'}
                                            </TableCell>
                                            <TableCell align="center">
                                                <Button
                                                    variant="text"
                                                    size="small"
                                                    startIcon={<EyeIcon sx={{ fontSize: '1rem' }} />}
                                                    onClick={() => handleViewMore(model)}
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
                                    ))}
                                    {filteredModels.length === 0 && (
                                        <TableRow>
                                            <TableCell colSpan={7} align="center" sx={{ py: 4 }}>
                                                No models registered yet.
                                            </TableCell>
                                        </TableRow>
                                    )}
                                </TableBody>
                            </Table>
                        </TableContainer>
                    )}
                </CustomTabPanel>

                <CustomTabPanel value={tabValue} index={1}>
                    <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center', p: 4, bgcolor: '#fbfbfb' }}>
                        <Paper
                            elevation={0}
                            sx={{
                                width: '100%',
                                maxWidth: '1000px',
                                border: '1px solid #E8E8E8',
                                borderRadius: '12px',
                                overflow: 'hidden'
                            }}
                        >
                            {/* Header */}
                            <Box sx={{ p: 2, borderBottom: '1px solid #E8E8E8', bgcolor: '#fff' }}>
                                <Typography sx={{ fontSize: '1.25rem', fontWeight: 600, color: '#131313', mb: 0.5 }}>
                                    API Key Configuration
                                </Typography>
                                <Typography sx={{ fontSize: '0.85rem', color: '#666D73' }}>
                                    Add a new AI model to the system registry
                                </Typography>
                            </Box>

                            {/* Body */}
                            <Box sx={{ p: 2, bgcolor: '#fff' }}>
                                <Grid container spacing={3}>
                                    <Grid size={{ xs: 12 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Display Name<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            placeholder="Enter Display Name"
                                            value={formData.name}
                                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Description<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            multiline
                                            rows={4}
                                            placeholder="Write Description"
                                            value={formData.description}
                                            onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Model Type<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            select
                                            fullWidth
                                            size="small"
                                            value={formData.model_type}
                                            onChange={(e) => setFormData({ ...formData, model_type: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        >
                                            <MenuItem value="api">API</MenuItem>
                                            <MenuItem value="ollama">Ollama</MenuItem>
                                            <MenuItem value="huggingface">HuggingFace</MenuItem>
                                            <MenuItem value="uploaded">Uploaded</MenuItem>
                                        </TextField>
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Provider<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            select
                                            fullWidth
                                            size="small"
                                            value={formData.provider}
                                            onChange={(e) => setFormData({ ...formData, provider: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        >
                                            <MenuItem value="openai">OpenAI</MenuItem>
                                            <MenuItem value="anthropic">Anthropic</MenuItem>
                                            <MenuItem value="ollama">Ollama</MenuItem>
                                            <MenuItem value="huggingface">HuggingFace</MenuItem>
                                        </TextField>
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Model ID/Name<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            placeholder="Enter Model ID/Name"
                                            value={formData.model_id}
                                            onChange={(e) => setFormData({ ...formData, model_id: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            API Key (optional)
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            type="password"
                                            placeholder="Enter API Key"
                                            value={formData.api_key}
                                            onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Custom Endpoint URL (optional)
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            placeholder="Enter Custom Endpoint URL"
                                            value={formData.custom_endpoint}
                                            onChange={(e) => setFormData({ ...formData, custom_endpoint: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Max Tokens<span style={{ color: 'red' }}>*</span>
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            placeholder="Enter Max Tokens"
                                            value={formData.max_tokens}
                                            onChange={(e) => setFormData({ ...formData, max_tokens: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12, md: 6 }}>
                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                            Tags (comma-separated)
                                        </Typography>
                                        <TextField
                                            fullWidth
                                            size="small"
                                            placeholder="Write Tags"
                                            value={formData.tags}
                                            onChange={(e) => setFormData({ ...formData, tags: e.target.value })}
                                            sx={{ '& .MuiOutlinedInput-root': { borderRadius: 2, fontSize: '0.9rem', '& fieldset': { borderColor: '#E8E8E8' } } }}
                                        />
                                    </Grid>

                                    <Grid size={{ xs: 12 }}>
                                        <Box sx={{ mt: 1 }}>
                                            <Typography sx={{ fontSize: '0.8rem', fontWeight: 500, mb: 1, color: '#313131' }}>
                                                Temperature: {formData.temperature}
                                            </Typography>
                                            <Slider
                                                value={formData.temperature}
                                                step={0.1}
                                                min={0}
                                                max={1}
                                                onChange={(_, val) => setFormData({ ...formData, temperature: val as number })}
                                                valueLabelDisplay="auto"
                                                sx={{ color: '#6366F1' }}
                                            />
                                        </Box>
                                    </Grid>

                                    <Grid size={{ xs: 12 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                                            <Button
                                                variant="contained"
                                                onClick={handleRegister}
                                                disabled={submitting || !formData.name || !formData.model_id || !formData.description}
                                                sx={{
                                                    textTransform: 'none',
                                                    bgcolor: '#6366F1',
                                                    color: '#fff',
                                                    borderRadius: '6px',
                                                    boxShadow: 'none',
                                                    fontWeight: 500,
                                                    px: 4,
                                                    py: 1,
                                                    fontSize: '0.875rem',
                                                    '&:hover': { bgcolor: '#4F46E5', boxShadow: 'none' },
                                                    '&:disabled': { bgcolor: '#E2E8F0', color: '#94A3B8' }
                                                }}
                                            >
                                                {submitting ? <CircularProgress size={20} color="inherit" /> : 'Register'}
                                            </Button>
                                        </Box>
                                    </Grid>
                                </Grid>
                            </Box>
                        </Paper>
                    </Box>
                </CustomTabPanel>

                <CustomTabPanel value={tabValue} index={2}>
                    <Box sx={{ p: 4, bgcolor: '#fbfbfb', minHeight: '600px' }}>
                        <Grid container spacing={3}>
                            {/* Summary Metrics */}
                            <Grid size={{ xs: 12, md: 3 }}>
                                <Paper sx={{ p: 3, borderRadius: 3, border: '1px solid #E8E8E8', boxShadow: 'none' }}>
                                    <Typography variant="body2" sx={{ color: '#666D73', mb: 1 }}>Total Models</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#131313' }}>{models.length}</Typography>
                                    <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <Typography sx={{ color: '#10b981', fontSize: '0.8rem', fontWeight: 600 }}>+{models.filter(m => m.status === 'active').length}</Typography>
                                        <Typography sx={{ color: '#666D73', fontSize: '0.8rem' }}>Active now</Typography>
                                    </Box>
                                </Paper>
                            </Grid>
                            <Grid size={{ xs: 12, md: 3 }}>
                                <Paper sx={{ p: 3, borderRadius: 3, border: '1px solid #E8E8E8', boxShadow: 'none' }}>
                                    <Typography variant="body2" sx={{ color: '#666D73', mb: 1 }}>Avg Latency</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#131313' }}>
                                        {(models.reduce((acc, m) => acc + (m.avg_latency_ms || 0), 0) / (models.length || 1)).toFixed(2)}ms
                                    </Typography>
                                    <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <Typography sx={{ color: '#6366F1', fontSize: '0.8rem', fontWeight: 600 }}>Target: &lt;200ms</Typography>
                                    </Box>
                                </Paper>
                            </Grid>
                            <Grid size={{ xs: 12, md: 3 }}>
                                <Paper sx={{ p: 3, borderRadius: 3, border: '1px solid #E8E8E8', boxShadow: 'none' }}>
                                    <Typography variant="body2" sx={{ color: '#666D73', mb: 1 }}>Total API Calls</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#131313' }}>
                                        {models.reduce((acc, m) => acc + (m.total_requests || 0), 0).toLocaleString()}
                                    </Typography>
                                    <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <Typography sx={{ color: '#666D73', fontSize: '0.8rem' }}>Last 30 days</Typography>
                                    </Box>
                                </Paper>
                            </Grid>
                            <Grid size={{ xs: 12, md: 3 }}>
                                <Paper sx={{ p: 3, borderRadius: 3, border: '1px solid #E8E8E8', boxShadow: 'none' }}>
                                    <Typography variant="body2" sx={{ color: '#666D73', mb: 1 }}>Total Tokens</Typography>
                                    <Typography variant="h4" sx={{ fontWeight: 700, color: '#131313' }}>
                                        {(models.reduce((acc, m) => acc + (m.total_tokens || 0), 0) / 1000).toFixed(1)}k
                                    </Typography>
                                    <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                        <Typography sx={{ color: '#666D73', fontSize: '0.8rem' }}>Consolidated</Typography>
                                    </Box>
                                </Paper>
                            </Grid>

                            {/* Models by Provider Chart */}
                            <Grid size={{ xs: 12, md: 6 }}>
                                <Paper sx={{ p: 4, borderRadius: 3, border: '1px solid #E8E8E8', boxShadow: 'none', height: '400px' }}>
                                    <Typography sx={{ fontWeight: 600, mb: 3 }}>Models by Provider</Typography>
                                    <ResponsiveContainer width="100%" height="90%">
                                        <PieChart>
                                            <Tooltip
                                                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                            />
                                            <Legend verticalAlign="bottom" height={36} />
                                            <Pie
                                                data={Array.from(new Set(models.map(m => m.provider))).map(provider => ({
                                                    name: provider,
                                                    value: models.filter(m => m.provider === provider).length
                                                }))}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="value"
                                                nameKey="name"
                                            >
                                                {Array.from(new Set(models.map(m => m.provider))).map((_, index) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                        </PieChart>
                                    </ResponsiveContainer>
                                </Paper>
                            </Grid>

                            {/* Usage Distribution Chart */}
                            <Grid size={{ xs: 12, md: 6 }}>
                                <Paper sx={{ p: 4, borderRadius: 3, border: '1px solid #E8E8E8', boxShadow: 'none', height: '400px' }}>
                                    <Typography sx={{ fontWeight: 600, mb: 3 }}>Usage Distribution (Requests)</Typography>
                                    <ResponsiveContainer width="100%" height="90%">
                                        <PieChart>
                                            <Tooltip
                                                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                            />
                                            <Legend verticalAlign="bottom" height={36} />
                                            <Pie
                                                data={models.filter(m => (m.total_requests || 0) > 0)}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="total_requests"
                                                nameKey="name"
                                            >
                                                {models.map((_, index) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                                ))}
                                            </Pie>
                                        </PieChart>
                                    </ResponsiveContainer>
                                </Paper>
                            </Grid>

                            {/* Token Distribution Chart */}
                            <Grid size={{ xs: 12, md: 6 }}>
                                <Paper sx={{ p: 4, borderRadius: 3, border: '1px solid #E8E8E8', boxShadow: 'none', height: '400px' }}>
                                    <Typography sx={{ fontWeight: 600, mb: 3 }}>Token Distribution</Typography>
                                    <ResponsiveContainer width="100%" height="90%">
                                        <PieChart>
                                            <Tooltip
                                                contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                            />
                                            <Legend verticalAlign="bottom" height={36} />
                                            <Pie
                                                data={models.filter(m => (m.total_tokens || 0) > 0)}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="total_tokens"
                                                nameKey="name"
                                            >
                                                {models.map((_, index) => (
                                                    <Cell key={`cell-${index}`} fill={COLORS[(index + 2) % COLORS.length]} />
                                                ))}
                                            </Pie>
                                        </PieChart>
                                    </ResponsiveContainer>
                                </Paper>
                            </Grid>
                        </Grid>
                    </Box>
                </CustomTabPanel>
            </Paper>

            <Dialog
                open={detailsOpen}
                onClose={() => setDetailsOpen(false)}
                maxWidth="md"
                fullWidth
                PaperProps={{ sx: { borderRadius: 3, boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)' } }}
            >
                <DialogTitle sx={{ p: 4, pb: 2, borderBottom: '1px solid #F1F5F9' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
                                <Typography variant="h5" sx={{ fontWeight: 700, color: '#131313' }}>{selectedModel?.name}</Typography>
                                <Chip
                                    label={selectedModel?.status === 'active' ? 'Active' : 'Inactive'}
                                    size="small"
                                    sx={{
                                        bgcolor: selectedModel?.status === 'active' ? '#DCFCE7' : '#FEE2E2',
                                        color: selectedModel?.status === 'active' ? '#166534' : '#991B1B',
                                        fontWeight: 600,
                                        borderRadius: 1.5,
                                        height: 24
                                    }}
                                />
                            </Box>
                            <Typography sx={{ color: '#666D73', fontSize: '0.9rem' }}>
                                {selectedModel ? modelDescriptions[selectedModel.name] || 'AI model; API key required' : ''}
                            </Typography>
                            <Box sx={{ display: 'flex', gap: 1, mt: 2 }}>
                                {['LLM', selectedModel?.provider || 'OpenAI', 'Production'].map(tag => (
                                    <Chip key={tag} label={tag} size="small" variant="outlined" sx={{ border: 'none', borderRadius: 1.5, backgroundColor: '#F5F5F5', color: '#1313133', borderColor: '#E2E8F0', height: 28, fontSize: '0.75rem' }} />
                                ))}
                            </Box>
                            {/* alpha('#6366F1', 0.1) */}
                        </Box>
                        <IconButton onClick={() => setDetailsOpen(false)} sx={{ fontSize: '10px', bgcolor: '#6366F1', color: '#FFFFFF', '&:hover': { bgcolor: alpha('#6366F1', 0.1) }, borderRadius: '20% !important', p: '4px !important' }}>
                            <CloseIcon sx={{ fontSize: '0.8rem !important' }} />
                        </IconButton>
                    </Box>
                </DialogTitle>
                <DialogContent sx={{ p: 4 }}>
                    <Grid container spacing={2} sx={{ mb: 4 }}>
                        <Grid size={{ xs: 12, sm: 4 }}>
                            {/* <InfoCard icon={<CategoryOutlined sx={{ color: '#6366F1' }} />} label="Type" value={selectedModel?.model_type || 'api'} /> */}
                            <InfoCard icon={<img src="./link-circle.png" />} label="Type" value={selectedModel?.model_type || 'api'} />
                        </Grid>
                        <Grid size={{ xs: 12, sm: 4 }}>
                            <InfoCard icon={<img src="./provider.png" />} label="Provider" value={selectedModel?.provider || 'OpenAI'} />
                        </Grid>
                        <Grid size={{ xs: 12, sm: 4 }}>
                            <InfoCard icon={<img src="./temperature.png" />} label="Temperature" value={selectedModel?.temperature || '0.7'} />
                        </Grid>
                        <Grid size={{ xs: 12, sm: 4 }}>
                            <InfoCard icon={<img src="./token.png" />} label="Max Tokens" value={selectedModel?.max_tokens?.toLocaleString() || '4,096'} />
                        </Grid>
                        <Grid size={{ xs: 12, sm: 4 }}>
                            <InfoCard icon={<img src="./timer.png" />} label="Total Request" value={selectedModel?.total_requests || '0'} />
                        </Grid>
                        <Grid size={{ xs: 12, sm: 4 }}>
                            <InfoCard icon={<img src="./latency.png" />} label="Avg Latency" value={selectedModel?.avg_latency_ms ? `${selectedModel.avg_latency_ms.toFixed(2)} ms` : '0.00 ms'} />
                        </Grid>
                    </Grid>

                    <Box sx={{ borderTop: '1px solid #F1F5F9', pt: 3 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                            <Typography sx={{ fontWeight: 700, fontSize: '1.1rem', color: '#131313' }}>API Key Configuration</Typography>
                            <Chip
                                label={selectedModel?.api_key ? 'Configured' : 'Not configured'}
                                size="small"
                                sx={{
                                    bgcolor: selectedModel?.api_key ? alpha('#10b981', 0.1) : '#FEF3C7',
                                    color: selectedModel?.api_key ? '#059669' : '#D97706',
                                    fontWeight: 400,
                                    borderRadius: 1.5
                                }}
                            />
                        </Box>
                        <Typography sx={{ fontSize: '0.85rem', color: '#666D73', mb: 1 }}>Update API Key</Typography>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                            <TextField
                                fullWidth
                                size="small"
                                placeholder="Enter New API Key........"
                                sx={{
                                    '& .MuiOutlinedInput-root': {
                                        borderRadius: 2,
                                        fontSize: '0.9rem',
                                        bgcolor: '#fff',
                                        '& fieldset': { borderColor: '#E2E8F0' },
                                        '&:hover fieldset': { borderColor: '#6366F1' },
                                    }
                                }}
                                InputProps={{
                                    endAdornment: <EyeIcon sx={{ color: '#94A3B8', fontSize: 20, cursor: 'pointer' }} />
                                }}
                            />
                            <Button
                                variant="contained"
                                sx={{
                                    bgcolor: '#6366F1',
                                    minWidth: 48,
                                    width: 48,
                                    p: 0,
                                    borderRadius: 2,
                                    boxShadow: 'none',
                                    '&:hover': { bgcolor: '#4F46E5', boxShadow: 'none' }
                                }}
                            >
                                <CheckCircleIcon sx={{ fontSize: 20 }} />
                            </Button>
                        </Box>
                    </Box>
                </DialogContent>
                <DialogActions sx={{ p: 4, pt: 0, justifyContent: 'space-between' }}>
                    <Button
                        variant="outlined"
                        sx={{
                            textTransform: 'none',
                            color: '#6366F1',
                            borderColor: '#6366F1',
                            borderRadius: 2,
                            px: 3,
                            fontWeight: 600,
                            '&:hover': { bgcolor: alpha('#6366F1', 0.04), borderColor: '#4F46E5' }
                        }}
                    >
                        Delete
                    </Button>
                    <Box sx={{ display: 'flex', gap: 2 }}>
                        <Button
                            variant="outlined"
                            sx={{
                                textTransform: 'none',
                                color: '#6366F1',
                                borderColor: '#6366F1',
                                borderRadius: 2,
                                px: 3,
                                fontWeight: 600,
                                '&:hover': { bgcolor: alpha('#6366F1', 0.04), borderColor: '#4F46E5' }
                            }}
                        >
                            Test
                        </Button>
                        <Button
                            variant="contained"
                            sx={{
                                textTransform: 'none',
                                bgcolor: '#6366F1',
                                borderRadius: 2,
                                px: 3,
                                fontWeight: 600,
                                boxShadow: 'none',
                                '&:hover': { bgcolor: '#4F46E5', boxShadow: 'none' }
                            }}
                        >
                            Active
                        </Button>
                    </Box>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default Model;
