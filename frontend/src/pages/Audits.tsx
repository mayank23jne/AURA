import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Paper,
    Button,
    Grid,
    Card,
    CardContent,
    CircularProgress,
    Tabs,
    Tab,
    Slider,
    Tooltip,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Chip,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    IconButton
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CheckCircleOutlinedIcon from '@mui/icons-material/CheckCircleOutlined';
import ErrorOutlineOutlinedIcon from '@mui/icons-material/ErrorOutlineOutlined';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdfOutlined';
import CodeIcon from '@mui/icons-material/CodeOutlined';
import DescriptionIcon from '@mui/icons-material/DescriptionOutlined';
import VisibilityIcon from '@mui/icons-material/VisibilityOutlined';
import CloseIcon from '@mui/icons-material/Close';
import { api } from '../api/client';
import { alpha } from '@mui/material/styles';
import type { Policy, Recommendation } from '../Types';

const steps = ['Select Model', 'Choose Framework', 'Select Compliance', 'Internal Controls', 'Audit Results'];

const Audits: React.FC = () => {
    // Print styles
    useEffect(() => {
        const style = document.createElement('style');
        style.innerHTML = `
            @media print {
                .no-print, .MuiTabs-root, .MuiButton-root, .MuiStepper-root, .MuiTab-root { display: none !important; }
                body { background: white !important; margin: 0; padding: 0; }
                .MuiPaper-root { border: none !important; box-shadow: none !important; padding: 0 !important; }
                .print-content { width: 100% !important; margin: 0 !important; padding: 20px !important; }
                .MuiGrid-item { page-break-inside: avoid; }
                .MuiTable-root { page-break-inside: auto; }
                tr { page-break-inside: avoid; page-break-after: auto; }
                thead { display: table-header-group; }
                tfoot { display: table-footer-group; }
            }
        `;
        document.head.appendChild(style);
        return () => { document.head.removeChild(style); };
    }, []);

    const [activeTab, setActiveTab] = useState(0);
    const [activeStep, setActiveStep] = useState(0);
    const [models, setModels] = useState<any[]>([]);
    const [policies, setPolicies] = useState<Policy[]>([]);
    const [selectedModel, setSelectedModel] = useState<any | null>(null);
    const [selectedFrameworks, setSelectedFrameworks] = useState<string[]>(['aura-native']);
    const [selectedPolicies, setSelectedPolicies] = useState<string[]>([]);
    const [selectedControls, setSelectedControls] = useState<string[]>([]);
    const [testsPerPolicy, setTestsPerPolicy] = useState<number>(30);
    const [loading, setLoading] = useState(true);
    const [running, setRunning] = useState(false);
    const [auditResults, setAuditResults] = useState<any>(null);
    const [auditProgress, setAuditProgress] = useState(0);
    const [auditStage, setAuditStage] = useState('Initializing');
    const [auditLogs, setAuditLogs] = useState<string[]>([]);
    const [startTime, setStartTime] = useState<Date | null>(null);
    const [reportTab, setReportTab] = useState(0);
    const [rawModalOpen, setRawModalOpen] = useState(false);

    useEffect(() => {
        const mockModels = [
            { id: 'gpt4', name: 'GPT-4', provider: 'OpenAI GPT-4 model - requires OPENAI_API_KEY environment variable', active: true },
            { id: 'gpt35', name: 'GPT-3.5', provider: 'OpenAI GPT-3.5 model - requires OPENAI_API_KEY environment variable', active: true },
            { id: 'claude2', name: 'Claude 2', provider: 'Anthropic Claude 2 model - requires AUTH_TOKEN environment variable', active: true },
            { id: 'llama', name: 'LLaMA', provider: 'Meta LLaMA model - requires ACCESS_TOKEN environment variable', active: true },
        ];

        const fetchModels = async () => {
            try {
                const response = await api.getModels();
                if (response.data && response.data.models && response.data.models.length > 0) {
                    setModels(response.data.models.map((m: any) => ({ ...m, active: true })));
                    setSelectedModel(response.data.models[0]);
                } else {
                    setModels(mockModels);
                    setSelectedModel(mockModels[0]);
                }
            } catch (error) {
                console.error('Error fetching models:', error);
                setModels(mockModels);
                setSelectedModel(mockModels[0]);
            } finally {
                setLoading(false);
            }
        };
        fetchModels();

        const fetchPolicies = async () => {
            try {
                const response = await api.getPolicies();
                console.log('fetch policy', response.data);
                if (response.data && response.data.policies && response.data.policies.length > 0) {
                    setPolicies(response.data.policies.map((m: Policy) => ({ ...m, active: true })));
                    console.log('selected policies', response.data.policies[0]);
                    setSelectedPolicies(response.data.policies[0]);
                } else {
                    setPolicies(policies);
                    setSelectedPolicies([policies[0]?.id]);
                }
            } catch (error) {
                console.error('Error fetching policy:', error);
                setPolicies(policies);
                setSelectedPolicies([policies[0]?.id]);
            } finally {
                setLoading(false);
            }
        };
        fetchPolicies();
    }, []);

    const handleNext = () => setActiveStep((prev) => Math.min(prev + 1, steps.length - 1));
    const handleBack = () => setActiveStep((prev) => Math.max(prev - 1, 0));
    const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => setActiveTab(newValue);

    const handleStartAudit = async () => {
        if (!selectedModel) return;
        setRunning(true);
        setActiveStep(4);
        setAuditProgress(10);
        setAuditStage('Initializing');
        setStartTime(new Date());
        setAuditLogs(['Initializing audit context...', 'Loading configurations...']);

        try {
            // Stage 1: Initialize
            setAuditStage('Initializing');
            setAuditProgress(15);
            await new Promise(r => setTimeout(r, 800));

            // Stage 2: Load Policies
            setAuditStage('Loading Policies');
            setAuditProgress(30);
            setAuditLogs(prev => [...prev, `Loading ${selectedPolicies.length} selected policies...`]);
            await new Promise(r => setTimeout(r, 800));

            // Stage 3: Running Tests
            setAuditStage('Running Tests');
            setAuditProgress(45);
            setAuditLogs(prev => [...prev, `Starting test execution across ${selectedFrameworks.length} frameworks...`]);

            const response = await api.runAudit({
                model_id: selectedModel.id,
                frameworks: selectedFrameworks,
                policy_ids: selectedPolicies,
                test_count: testsPerPolicy,
            });

            // Stage 4: Analyze Results
            setAuditStage('Analyzing Results');
            setAuditProgress(75);
            setAuditLogs(prev => [...prev, 'Test execution finished. Parsing results...']);
            await new Promise(r => setTimeout(r, 1000));

            // Stage 5: Generate Report
            setAuditStage('Generating Report');
            setAuditProgress(95);
            setAuditLogs(prev => [...prev, 'Generating comprehensive compliance report...']);
            await new Promise(r => setTimeout(r, 800));

            // Complete
            console.log('auditresult', response.data)
            setAuditStage('Completed');
            setAuditProgress(100);
            setAuditResults(response.data);
            setAuditLogs(prev => [...prev, 'Audit completed successfully. Check the summary below.']);

        } catch (error: any) {
            console.error('Audit failed:', error);
            setAuditStage('Failed');
            setAuditLogs(prev => [...prev, `Critical Error: ${error.message || 'Audit failed'}`]);
        } finally {
            setRunning(false);
        }
    };

    const handleExportMarkdown = () => {
        if (!auditResults) return;

        let md = `# Audit Report - ${auditResults.audit_id || 'Summary'}\n\n`;
        md += `## Overview\n`;
        md += `- **Compliance Score:** ${(auditResults.compliance_score).toFixed(1)}%\n`;
        md += `- **Passed Tests:** ${auditResults.results?.passed_count || 0}\n`;
        md += `- **Total Tests:** ${auditResults.results?.total_tests || 0}\n\n`;

        md += `## Policy Breakdown\n`;
        Object.entries(auditResults.results?.by_policy || {}).forEach(([policy, data]: [string, any]) => {
            md += `- **${policy}:** ${(data.score * 100).toFixed(0)}%\n`;
        });

        md += `\n## Recommendations\n`;
        auditResults.recommendations?.forEach((rec: Recommendation) => {
            md += `### ${rec.title}\n`;
            md += `- **Priority:** ${rec.priority}\n`;
            md += `- **ID:** ${rec.finding_id}\n`;
            md += `${rec.description}\n\n`;
        });

        md += `\n## Findings\n`;
        (auditResults.findings || []).forEach((finding: any) => {
            md += `### [${finding.severity}] ${finding.title}\n`;
            md += `${finding.description}\n\n`;
        });

        const blob = new Blob([md], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `audit-report-${auditResults.audit_id || 'export'}.md`;
        a.click();
    };

    const handleExportPDF = async () => {
        if (!auditResults?.audit_id) return;
        try {
            const response = await api.getAuditPDF(auditResults.audit_id);
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `audit-report-${auditResults.audit_id}.pdf`);
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error('Export PDF failed:', error);
            // Fallback to print if API fails
            window.print();
        }
    };

    return (
        <Box sx={{ width: '100%', mt: -3, px: 0 }}>
            {/* Tab Navigation Layer */}
            <Box sx={{
                borderBottom: 1,
                borderColor: '#E5E7EB',
                backgroundColor: '#FFFFFF',
                mx: -2,
                px: 2,
                width: 'calc(100% + 32px)'
            }} className="no-print">
                <Tabs
                    value={activeTab}
                    onChange={handleTabChange}
                    sx={{
                        '& .MuiTab-root': {
                            textTransform: 'none',
                            fontWeight: 400,
                            fontSize: '0.85rem',
                            minWidth: 'auto',
                            mr: 4,
                            color: '#131313',
                            px: 0,
                            pb: 1,
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
                            }
                        },
                        '& .Mui-selected': {
                            color: '#6366f1 !important',
                            border: 'none',
                            outline: 'none',
                        },
                        '& .MuiTabs-indicator': {
                            backgroundColor: '#6366f1',
                            height: 1.5,
                        }
                    }}
                >
                    <Tab label="Wizard Mode" />
                    <Tab label="Audit History" />
                </Tabs>
            </Box>

            {activeTab === 0 && (
                <Box className={activeStep < 4 ? '' : 'print-content'}>
                    {/* Stepper Navigation Bar */}
                    <Box sx={{
                        backgroundColor: '#FFFFFF',
                        borderBottom: 1,
                        borderColor: '#E5E7EB',
                        mx: -2,
                        width: 'calc(100% + 32px)'
                    }} className="no-print">
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', p: 1, gap: 2 }}>
                            <Box sx={{
                                flex: 1,
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                                px: 1,
                                position: 'relative'
                            }}>
                                <Button
                                    onClick={handleBack}
                                    disabled={activeStep === 0}
                                    sx={{
                                        textTransform: 'none',
                                        color: activeStep === 0 ? '#CBCDCE' : '#64748b',
                                        fontSize: '0.8rem',
                                        fontWeight: 400,
                                        border: activeStep === 0 ? '1px solid #f1f5f9' : '1px solid #CBCDCE',
                                        borderRadius: '12px',
                                        px: 2,
                                        height: 30,
                                        minWidth: 'auto',
                                        '&:hover': { bgcolor: 'transparent', borderColor: '#cbd5e1' }
                                    }}
                                >
                                    Previous
                                </Button>

                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1, justifyContent: 'center', mx: 2 }}>
                                    {steps.map((label, index) => (
                                        <React.Fragment key={label}>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.8 }}>
                                                <Box sx={{
                                                    width: 18,
                                                    height: 18,
                                                    borderRadius: '50%',
                                                    bgcolor: index < activeStep ? '#6366f1' : (activeStep === index ? '#6366f1' : '#fff'),
                                                    color: index < activeStep ? '#fff' : (activeStep === index ? '#fff' : '#94a3b8'),
                                                    border: activeStep === index ? '1px solid #6366f1' : (index < activeStep ? '1px solid #6366f1' : '1px solid #e2e8f0'),
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    fontSize: '0.6rem',
                                                    fontWeight: 600
                                                }}>
                                                    {index < activeStep ? (
                                                        <CheckCircleIcon sx={{ fontSize: 12 }} />
                                                    ) : (
                                                        index + 1
                                                    )}
                                                </Box>
                                                <Typography sx={{
                                                    fontSize: '0.65rem',
                                                    fontWeight: 400,
                                                    whiteSpace: 'nowrap',
                                                    color: activeStep === index ? '#6366f1' : (index < activeStep ? '#6366f1' : '#A4A6A8')
                                                }}>
                                                    {label}
                                                </Typography>
                                            </Box>
                                            {index < steps.length - 1 && (
                                                <Box sx={{
                                                    width: 30,
                                                    height: 0,
                                                    borderTop: '1px dotted #A5B4FC',
                                                    mx: 0.4
                                                }} />
                                            )}
                                        </React.Fragment>
                                    ))}
                                </Box>

                                <Button
                                    onClick={handleNext}
                                    disabled={activeStep === steps.length - 1}
                                    sx={{
                                        textTransform: 'none',
                                        color: activeStep === steps.length - 1 ? '#CBCDCE' : '#131313',
                                        fontSize: '0.8rem',
                                        fontWeight: 500,
                                        border: activeStep === steps.length - 1 ? '1px solid #f1f5f9' : '1px solid #CBCDCE',
                                        borderRadius: '12px',
                                        px: 2,
                                        height: 30,
                                        minWidth: 'auto',
                                        '&:hover': { bgcolor: 'transparent', borderColor: '#cbd5e1' }
                                    }}
                                >
                                    Next
                                </Button>
                            </Box>


                        </Box>
                    </Box>

                    {/* Step Content Card - Full Width */}
                    <Box sx={{ width: '100%', mt: 2, px: 3 }}>
                        <Paper
                            elevation={0}
                            sx={{
                                borderRadius: 3,
                                border: '1px solid #f1f5f9',
                                bgcolor: '#fff',
                                mb: 8
                            }}
                        >
                            {activeStep === 0 && (
                                <Box>
                                    <Box sx={{ borderBottom: '1px solid #E8E8E8', p: 1 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 500, fontSize: '20px', color: '#131313', mb: 0.8 }}>
                                            Select the AI Model to Audit
                                        </Typography>
                                        <Typography variant="body2" sx={{ color: '#666D73', mb: 1, fontSize: '0.85rem' }}>
                                            Choose which model you want to test for security, bias, and compliance.
                                        </Typography>
                                    </Box>

                                    <Grid container spacing={1} mt={2} mb={2} p={1} sx={{ borderBottom: '1px solid #E8E8E8' }}>
                                        {loading ? (
                                            <Box sx={{ display: 'flex', justifyContent: 'center', width: '100%', py: 4 }}><CircularProgress size={24} /></Box>
                                        ) : (
                                            models.map((m) => (
                                                <Grid size={{ xs: 12, sm: 6, md: 6 }} key={m.id}>
                                                    <Card
                                                        onClick={() => setSelectedModel(m)}
                                                        elevation={0}
                                                        sx={{
                                                            cursor: 'pointer',
                                                            borderRadius: 2.5,
                                                            border: selectedModel?.id === m.id ? '1px solid #6366f1' : '1px solid #f1f5f9',
                                                            bgcolor: selectedModel?.id === m.id ? alpha('#6366f1', 0.02) : '#fff',
                                                            transition: 'all 0.1s ease-in-out',
                                                            height: '100%',
                                                            '&:hover': {
                                                                borderColor: selectedModel?.id === m.id ? '#6366f1' : '#cbd5e1',
                                                            }
                                                        }}
                                                    >
                                                        <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                                                            <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.8, gap: 1.2 }}>
                                                                <Typography sx={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>
                                                                    {m.name}
                                                                </Typography>
                                                                {m.status === 'active' ? <Box sx={{
                                                                    bgcolor: '#dcfce7',
                                                                    color: '#16a34a', // Pure green
                                                                    px: 1.2,
                                                                    py: 0.4,
                                                                    borderRadius: 1,
                                                                    fontSize: '0.65rem',
                                                                    fontWeight: 400
                                                                }}>
                                                                    Active
                                                                </Box> : <Box sx={{
                                                                    bgcolor: '#fcdcdeff',
                                                                    color: '#a31622ff', // Pure green
                                                                    px: 1.2,
                                                                    py: 0.4,
                                                                    borderRadius: 1,
                                                                    fontSize: '0.65rem',
                                                                    fontWeight: 400
                                                                }}>
                                                                    Inactive
                                                                </Box>}
                                                            </Box>
                                                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem', lineHeight: 1.6 }}>
                                                                {/* {m.provider} */}
                                                                OpenAI GPT-4 model - requires OPENAI_API_KEY environment variable
                                                            </Typography>
                                                        </CardContent>
                                                    </Card>
                                                </Grid>
                                            ))
                                        )}
                                    </Grid>
                                    {/* Action Buttons */}
                                    <Box sx={{ mt: 2, p: 2, display: 'flex', justifyContent: 'space-between' }}>
                                        <Button
                                            variant="outlined"
                                            size="small"
                                            onClick={handleBack}
                                            disabled={activeStep === 0}
                                            sx={{
                                                textTransform: 'none',
                                                color: '#64748b',
                                                borderColor: '#e2e8f0',
                                                borderRadius: 2,
                                                fontWeight: 600,
                                                px: 3,
                                                py: 1,
                                                fontSize: '0.85rem'
                                            }}
                                        >
                                            Previous
                                        </Button>

                                        <Button
                                            variant="contained"
                                            size="small"
                                            onClick={activeStep === steps.length - 1 ? handleStartAudit : handleNext}
                                            sx={{
                                                textTransform: 'none',
                                                bgcolor: '#6366f1',
                                                borderRadius: 2,
                                                fontWeight: 600,
                                                px: 3.5,
                                                py: 1,
                                                boxShadow: 'none',
                                                fontSize: '0.85rem',
                                                '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' }
                                            }}
                                        >
                                            {activeStep === steps.length - 1 ? 'Start Audit' : 'Save & Next'}
                                        </Button>
                                    </Box>
                                </Box>
                            )}

                            {activeStep === 1 && (
                                <Box>
                                    <Box sx={{ borderBottom: '1px solid #E8E8E8', p: 1 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 500, fontSize: '20px', color: '#131313', mb: 0.8 }}>
                                            Choose Testing Frameworks
                                        </Typography>
                                        <Typography variant="body2" sx={{ color: '#666D73', mb: 1, fontSize: '0.85rem' }}>
                                            Select one or more frameworks to run comprehensive security tests.
                                        </Typography>
                                    </Box>

                                    <Grid container spacing={2} mt={2} mb={2} p={1} sx={{ borderBottom: '1px solid #E8E8E8' }}>
                                        {[
                                            { id: 'aura-native', name: 'AURA Native', desc: 'Built-in testing framework with comprehensive test coverage..' },
                                            { id: 'garak', name: 'Garak', desc: 'LLM vulnerability scanner by NVIDIA' },
                                            { id: 'pyrit', name: 'PyRIT', desc: "Microsoft's AI Red Team framework" },
                                            { id: 'aiverify', name: 'AI Verify', desc: "Singapore's AI governance framework for supervised learning models" }
                                        ].map((f) => (
                                            <Grid size={{ xs: 12, sm: 6, md: 6 }} key={f.id}>
                                                <Card
                                                    onClick={() => {
                                                        setSelectedFrameworks(prev =>
                                                            prev.includes(f.id)
                                                                ? prev.filter(id => id !== f.id)
                                                                : [...prev, f.id]
                                                        );
                                                    }}
                                                    elevation={0}
                                                    sx={{
                                                        cursor: 'pointer',
                                                        borderRadius: 2.5,
                                                        border: selectedFrameworks.includes(f.id) ? '1px solid #6366f1' : '1px solid #f1f5f9',
                                                        bgcolor: selectedFrameworks.includes(f.id) ? alpha('#6366f1', 0.02) : '#fff',
                                                        transition: 'all 0.1s ease-in-out',
                                                        height: '100%',
                                                        '&:hover': {
                                                            borderColor: selectedFrameworks.includes(f.id) ? '#6366f1' : '#cbd5e1',
                                                        }
                                                    }}
                                                >
                                                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                                                        <Typography sx={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a', mb: 0.5 }}>
                                                            {f.name}
                                                        </Typography>
                                                        <Typography sx={{ color: '#64748b', fontSize: '0.8rem', lineHeight: 1.6 }}>
                                                            {f.desc}
                                                        </Typography>
                                                    </CardContent>
                                                </Card>
                                            </Grid>
                                        ))}
                                    </Grid>

                                    {/* Action Buttons */}
                                    <Box sx={{ mt: 2, p: 2, display: 'flex', justifyContent: 'space-between' }}>
                                        <Button
                                            variant="outlined"
                                            size="small"
                                            onClick={handleBack}
                                            sx={{
                                                textTransform: 'none',
                                                color: '#6366f1',
                                                border: '1px solid #6366f1',
                                                borderRadius: 2,
                                                fontWeight: 500,
                                                px: 3,
                                                py: 1,
                                                fontSize: '0.85rem',
                                                '&:hover': {
                                                    bgcolor: alpha('#6366f1', 0.05),
                                                    border: '1px solid #6366f1',
                                                }
                                            }}
                                        >
                                            Previous
                                        </Button>

                                        <Button
                                            variant="contained"
                                            size="small"
                                            onClick={handleNext}
                                            sx={{
                                                textTransform: 'none',
                                                bgcolor: '#6366f1',
                                                borderRadius: 2,
                                                fontWeight: 500,
                                                px: 3.5,
                                                py: 1,
                                                boxShadow: 'none',
                                                fontSize: '0.85rem',
                                                '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' }
                                            }}
                                        >
                                            Save & Next
                                        </Button>
                                    </Box>
                                </Box>
                            )}

                            {activeStep === 2 && (
                                <Box>
                                    <Box sx={{ borderBottom: '1px solid #E8E8E8', p: 1 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 500, fontSize: '20px', color: '#131313', mb: 0.8 }}>
                                            Select Compliance Policies
                                        </Typography>
                                        <Typography variant="body2" sx={{ color: '#666D73', mb: 1, fontSize: '0.85rem' }}>
                                            Choose which policies and regulations to test against.
                                        </Typography>
                                    </Box>

                                    <Grid container spacing={2} mt={2} mb={2} p={1} sx={{ borderBottom: '1px solid #E8E8E8' }}>
                                        {policies.map((p) => (
                                            console.log('policiesp', p),
                                            <Grid size={{ xs: 12, sm: 6, md: 6 }} key={p.id}>
                                                <Card
                                                    onClick={() => {
                                                        setSelectedPolicies(prev => {
                                                            if (!Array.isArray(prev)) return [p.id]; // safety guard

                                                            return prev.includes(p.id)
                                                                ? prev.filter(id => id !== p.id)
                                                                : [...prev, p.id];
                                                        });
                                                    }}
                                                    elevation={0}
                                                    sx={{
                                                        cursor: 'pointer',
                                                        borderRadius: 2.5,
                                                        border: Array.isArray(selectedPolicies) && selectedPolicies.includes(p.id) ? '1px solid #6366f1' : '1px solid #f1f5f9',
                                                        bgcolor: Array.isArray(selectedPolicies) && selectedPolicies.includes(p.id) ? alpha('#6366f1', 0.02) : '#fff',
                                                        transition: 'all 0.1s ease-in-out',
                                                        height: '100%',
                                                        '&:hover': {
                                                            borderColor: Array.isArray(selectedPolicies) && selectedPolicies.includes(p.id) ? '#6366f1' : '#cbd5e1',
                                                        }
                                                    }}
                                                >
                                                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                                                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.6, gap: 1.2 }}>
                                                            <Typography sx={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a' }}>
                                                                {p.name}
                                                            </Typography>
                                                            <Box sx={{
                                                                // bgcolor: '#fef3c7',
                                                                // color: '#92400e',
                                                                bgcolor:
                                                                    p.severity.toLowerCase() === "critical"
                                                                        ? "#fee2e2"
                                                                        : p.severity.toLowerCase() === "high"
                                                                            ? "#fef3c7"
                                                                            : p.severity.toLowerCase() === "medium"
                                                                                ? "#dbeafe"
                                                                                : "#dcfce7",
                                                                color:
                                                                    p.severity.toLowerCase() === "critical"
                                                                        ? "#991b1b"
                                                                        : p.severity.toLowerCase() === "high"
                                                                            ? "#92400e"
                                                                            : p.severity.toLowerCase() === "medium"
                                                                                ? "#1e40af"
                                                                                : "#166534",
                                                                px: 1.2,
                                                                py: 0.4,
                                                                borderRadius: 1,
                                                                fontSize: '0.65rem',
                                                                fontWeight: 400
                                                            }}>
                                                                {p.severity}
                                                            </Box>
                                                        </Box>
                                                        <Tooltip title={p.description} arrow>
                                                            <Typography sx={{ color: '#64748b', fontSize: '0.8rem', lineHeight: 1.6 }}>
                                                                {/* {p.description} */}
                                                                {p.description.length > 50 ? p.description.slice(0, 50) + "..." : p.description}
                                                            </Typography>
                                                        </Tooltip>
                                                    </CardContent>
                                                </Card>
                                            </Grid>
                                        ))}
                                    </Grid>


                                    {/* Action Buttons */}
                                    <Box sx={{ mt: 2, p: 2, display: 'flex', justifyContent: 'space-between' }}>
                                        <Button
                                            variant="outlined"
                                            size="small"
                                            onClick={handleBack}
                                            sx={{
                                                textTransform: 'none',
                                                color: '#6366f1',
                                                border: '1px solid #6366f1',
                                                borderRadius: 2,
                                                fontWeight: 500,
                                                px: 3,
                                                py: 1,
                                                fontSize: '0.85rem',
                                                '&:hover': {
                                                    bgcolor: alpha('#6366f1', 0.05),
                                                    border: '1px solid #6366f1',
                                                }
                                            }}
                                        >
                                            Previous
                                        </Button>

                                        <Button
                                            variant="contained"
                                            size="small"
                                            onClick={handleNext}
                                            sx={{
                                                textTransform: 'none',
                                                bgcolor: '#6366f1',
                                                borderRadius: 2,
                                                fontWeight: 500,
                                                px: 3.5,
                                                py: 1,
                                                boxShadow: 'none',
                                                fontSize: '0.85rem',
                                                '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' }
                                            }}
                                        >
                                            Save & Next
                                        </Button>
                                    </Box>
                                </Box>
                            )}

                            {activeStep === 3 && (
                                <Box>
                                    <Box sx={{ borderBottom: '1px solid #E8E8E8', p: 1 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 500, fontSize: '20px', color: '#131313', mb: 0.8 }}>
                                            Implementing Internal Controls
                                        </Typography>
                                        <Typography variant="body2" sx={{ color: '#666D73', mb: 1, fontSize: '0.85rem' }}>
                                            Configure safety controls and mitigation strategies for the selected model.
                                        </Typography>
                                    </Box>

                                    {/* Slider Section */}
                                    <Box sx={{ px: 2, pb: 4, pt: 2 }}>
                                        <Typography sx={{ fontSize: '0.85rem', fontWeight: 500, color: '#131313', mb: 3 }}>
                                            Tests per Policy
                                        </Typography>
                                        <Box sx={{ position: 'relative', px: 1 }}>
                                            <Slider
                                                value={testsPerPolicy}
                                                onChange={(_, val) => setTestsPerPolicy(val as number)}
                                                min={0}
                                                max={50}
                                                step={1}
                                                sx={{
                                                    color: '#6366f1',
                                                    height: 4,
                                                    padding: '13px 0',
                                                    '& .MuiSlider-thumb': {
                                                        width: 10,
                                                        height: 10,
                                                        backgroundColor: '#6366f1',
                                                        '&:before': { display: 'none' },
                                                        '&:after': {
                                                            content: `"${testsPerPolicy}"`,
                                                            position: 'absolute',
                                                            top: -20,
                                                            fontSize: '0.75rem',
                                                            fontWeight: 600,
                                                            color: '#6366f1'
                                                        }
                                                    },
                                                    '& .MuiSlider-track': {
                                                        bgcolor: '#6366f1',
                                                        border: 'none',
                                                    },
                                                    '& .MuiSlider-rail': {
                                                        opacity: 0.2,
                                                        bgcolor: '#6366f1',
                                                    },
                                                }}
                                            />
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: -1 }}>
                                                <Typography sx={{ fontSize: '0.7rem', color: '#94a3b8' }}>0</Typography>
                                                <Typography sx={{ fontSize: '0.7rem', color: '#94a3b8' }}>50</Typography>
                                            </Box>
                                        </Box>
                                    </Box>

                                    {/* <Grid container spacing={2} mt={2} mb={2} p={1} sx={{ borderBottom: '1px solid #E8E8E8' }}>
                                        {[
                                            { id: 'input-filtering', name: 'Input Filtering', desc: 'Analyze and block potentially harmful user prompts before they reach the model.' },
                                            { id: 'output-scanning', name: 'Output Scanning', desc: 'Scan model responses for sensitive data, toxicity, or non-compliance.' },
                                            { id: 'rate-limiting', name: 'Context-Aware Rate Limiting', desc: 'Dynamically adjust rate limits based on user risk profiles and behavior.' },
                                            { id: 'auditing', name: 'Real-time Audit Logging', desc: 'Maintain comprehensive logs of all model interactions for auditing and forensic analysis.' }
                                        ].map((c) => (
                                            <Grid size={{ xs: 12, sm: 6, md: 6 }} key={c.id}>
                                                <Card
                                                    onClick={() => {
                                                        setSelectedControls(prev =>
                                                            prev.includes(c.id)
                                                                ? prev.filter(id => id !== c.id)
                                                                : [...prev, c.id]
                                                        );
                                                    }}
                                                    elevation={0}
                                                    sx={{
                                                        cursor: 'pointer',
                                                        borderRadius: 2.5,
                                                        border: selectedControls.includes(c.id) ? '1px solid #6366f1' : '1px solid #f1f5f9',
                                                        bgcolor: selectedControls.includes(c.id) ? alpha('#6366f1', 0.02) : '#fff',
                                                        transition: 'all 0.1s ease-in-out',
                                                        height: '100%',
                                                        '&:hover': {
                                                            borderColor: selectedControls.includes(c.id) ? '#6366f1' : '#cbd5e1',
                                                        }
                                                    }}
                                                >
                                                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                                                        <Typography sx={{ fontWeight: 700, fontSize: '1.05rem', color: '#0f172a', mb: 0.5 }}>
                                                            {c.name}
                                                        </Typography>
                                                        <Typography sx={{ color: '#64748b', fontSize: '0.8rem', lineHeight: 1.6 }}>
                                                            {c.desc}
                                                        </Typography>
                                                    </CardContent>
                                                </Card>
                                            </Grid>
                                        ))}
                                    </Grid> */}

                                    {/* Action Buttons */}
                                    <Box sx={{ mt: 2, p: 2, display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #E8E8E8' }}>
                                        <Button
                                            variant="outlined"
                                            size="small"
                                            onClick={handleBack}
                                            sx={{
                                                textTransform: 'none',
                                                color: '#6366f1',
                                                border: '1px solid #6366f1',
                                                borderRadius: 2,
                                                fontWeight: 500,
                                                px: 3,
                                                py: 1,
                                                fontSize: '0.85rem',
                                                '&:hover': {
                                                    bgcolor: alpha('#6366f1', 0.05),
                                                    border: '1px solid #6366f1',
                                                }
                                            }}
                                        >
                                            Previous
                                        </Button>

                                        <Button
                                            variant="contained"
                                            size="small"
                                            onClick={handleStartAudit}
                                            disabled={running}
                                            sx={{
                                                textTransform: 'none',
                                                bgcolor: '#6366f1',
                                                borderRadius: 2,
                                                fontWeight: 500,
                                                px: 3.5,
                                                py: 1,
                                                boxShadow: 'none',
                                                fontSize: '0.85rem',
                                                '&:hover': { bgcolor: '#4f46e5', boxShadow: 'none' }
                                            }}
                                        >
                                            {running ? <CircularProgress size={20} color="inherit" /> : 'Initialize Audit'}
                                        </Button>
                                    </Box>
                                </Box>
                            )}
                            {activeStep === 4 && (
                                <Box sx={{ p: 4 }}>
                                    {/* Pipeline Stages */}
                                    <Box sx={{ mb: 6, position: 'relative' }}>
                                        <Typography variant="h6" sx={{ fontSize: '1.2rem', fontWeight: 600, mb: 4, color: '#131313' }}>
                                            Audit Pipeline
                                        </Typography>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', position: 'relative' }}>
                                            <Box sx={{ position: 'absolute', top: 24, left: '10%', right: '10%', height: 4, bgcolor: '#e2e8f0', zIndex: 0 }} />
                                            <Box sx={{
                                                position: 'absolute',
                                                top: 24,
                                                left: `${10}%`,
                                                width: `${Math.max(0, (auditProgress - 15) * 0.94)}%`,
                                                height: 4,
                                                bgcolor: '#10b981',
                                                zIndex: 1,
                                                transition: 'width 0.5s ease'
                                            }} />

                                            {['Initialize', 'Load Policies', 'Run Tests', 'Analyze', 'Report'].map((stage, idx) => {
                                                const stageSteps = [15, 30, 45, 75, 100];
                                                const isCompleted = auditProgress >= stageSteps[idx];
                                                const isActive = auditStage.includes(stage) || (auditProgress > (stageSteps[idx - 1] || 0) && auditProgress < stageSteps[idx]);

                                                return (
                                                    <Box key={stage} sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', zIndex: 2, flex: 1 }}>
                                                        <Box sx={{
                                                            width: 48,
                                                            height: 48,
                                                            borderRadius: '50%',
                                                            bgcolor: isCompleted ? '#10b981' : (isActive ? '#6366f1' : '#fff'),
                                                            border: '3px solid',
                                                            borderColor: isCompleted ? '#10b981' : (isActive ? '#6366f1' : '#e2e8f0'),
                                                            color: isCompleted || isActive ? '#fff' : '#94a3b8',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'center',
                                                            boxShadow: isActive ? '0 0 15px rgba(99, 102, 241, 0.4)' : 'none',
                                                            transition: 'all 0.3s ease'
                                                        }}>
                                                            {isCompleted ? <CheckCircleIcon sx={{ fontSize: 24 }} /> : (idx + 1)}
                                                        </Box>
                                                        <Typography sx={{ mt: 1.5, fontSize: '0.85rem', fontWeight: isActive ? 600 : 500, color: isActive ? '#6366f1' : (isCompleted ? '#131313' : '#94a3b8') }}>
                                                            {stage}
                                                        </Typography>
                                                    </Box>
                                                );
                                            })}
                                        </Box>
                                    </Box>

                                    {/* Progress Card */}
                                    <Paper elevation={0} sx={{ p: 4, borderRadius: '16px', bgcolor: '#fff', border: '1px solid #E8E8E8', mb: 6 }}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                            <Box>
                                                <Typography sx={{ fontSize: '1.5rem', fontWeight: 700, color: '#131313', display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                                    {running ? 'Audit in Progress' : (auditStage === 'Completed' ? 'Audit Completed' : (auditStage === 'Failed' ? 'Audit Failed' : 'Initializing...'))}
                                                    {running && <CircularProgress size={24} sx={{ color: '#6366f1' }} />}
                                                </Typography>
                                                <Typography sx={{ color: '#666D73', fontSize: '0.9rem' }}>
                                                    Audit ID: {auditResults?.audit_id || 'Generating...'}
                                                </Typography>
                                            </Box>
                                            <Typography sx={{ fontSize: '2rem', fontWeight: 800, color: '#6366f1' }}>{auditProgress}%</Typography>
                                        </Box>

                                        <LinearProgress
                                            variant="determinate"
                                            value={auditProgress}
                                            sx={{
                                                height: 12, borderRadius: 6, bgcolor: '#f1f5f9', mb: 4,
                                                '& .MuiLinearProgress-bar': { borderRadius: 6, bgcolor: auditStage === 'Failed' ? '#ef4444' : '#6366f1' }
                                            }}
                                        />

                                        <Grid container spacing={3}>
                                            <Grid size={{ xs: 12, sm: 3, md: 3 }}>
                                                <Box sx={{ p: 2, bgcolor: '#f8fafc', borderRadius: 2, border: '1px solid #f1f5f9' }}>
                                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', mb: 0.5 }}>Test Stage</Typography>
                                                    <Typography sx={{ fontSize: '1.1rem', fontWeight: 700, color: '#1e293b' }}>{auditStage}</Typography>
                                                </Box>
                                            </Grid>
                                            <Grid size={{ xs: 12, sm: 3, md: 3 }}>
                                                <Box sx={{ p: 2, bgcolor: '#f8fafc', borderRadius: 2, border: '1px solid #f1f5f9' }}>
                                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', mb: 0.5 }}>Passed Tests</Typography>
                                                    <Typography sx={{ fontSize: '1.1rem', fontWeight: 700, color: '#10b981' }}>{auditResults?.results?.passed_count || 0}</Typography>
                                                </Box>
                                            </Grid>
                                            <Grid size={{ xs: 12, sm: 3, md: 3 }}>
                                                <Box sx={{ p: 2, bgcolor: '#f8fafc', borderRadius: 2, border: '1px solid #f1f5f9' }}>
                                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', mb: 0.5 }}>Failed Tests</Typography>
                                                    <Typography sx={{ fontSize: '1.1rem', fontWeight: 700, color: '#ef4444' }}>{auditResults?.results ? (auditResults.results.total_tests - auditResults.results.passed_count) : 0}</Typography>
                                                </Box>
                                            </Grid>
                                            <Grid size={{ xs: 12, sm: 3, md: 3 }}>
                                                <Box sx={{ p: 2, bgcolor: '#f8fafc', borderRadius: 2, border: '1px solid #f1f5f9' }}>
                                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', textTransform: 'uppercase', mb: 0.5 }}>Compliance Score</Typography>
                                                    <Typography sx={{ fontSize: '1.1rem', fontWeight: 700, color: '#6366f1' }}>{auditResults?.compliance_score ? `${(auditResults.compliance_score).toFixed(1)}%` : '--'}</Typography>
                                                </Box>
                                            </Grid>
                                        </Grid>
                                    </Paper>

                                    {/* Dashboard Content */}
                                    {auditResults && !running && (
                                        <Box sx={{ mt: 2 }}>
                                            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                                                <Tabs
                                                    value={reportTab}
                                                    onChange={(_e, v) => setReportTab(v)}
                                                    sx={{
                                                        borderBottom: 1, borderColor: 'divider',
                                                        '& .MuiTabs-indicator': { bgcolor: '#6366f1' },
                                                        '& .MuiTab-root': { textTransform: 'none', fontWeight: 600, fontSize: '0.9rem' },
                                                        '& .Mui-selected': { color: '#6366f1 !important' }
                                                    }}
                                                >
                                                    <Tab label="Executive Summary" />
                                                    <Tab label="Framework Results" />
                                                    <Tab label="Findings & Risks" />
                                                    <Tab label="Raw Report" icon={<VisibilityIcon sx={{ fontSize: 16 }} />} iconPosition="start" />
                                                </Tabs>
                                                <Box sx={{ display: 'flex', gap: 1.5 }}>
                                                    <Tooltip title="Download PDF Report">
                                                        <Button
                                                            variant="outlined"
                                                            onClick={handleExportPDF}
                                                            startIcon={<PictureAsPdfIcon />}
                                                            size="small"
                                                            sx={{ textTransform: 'none', borderRadius: 2, color: '#ef4444', borderColor: '#ef4444', fontSize: '0.75rem' }}
                                                        >
                                                            PDF
                                                        </Button>
                                                    </Tooltip>
                                                    <Tooltip title="Export Markdown">
                                                        <Button
                                                            variant="outlined"
                                                            onClick={handleExportMarkdown}
                                                            startIcon={<DescriptionIcon />}
                                                            size="small"
                                                            sx={{ textTransform: 'none', borderRadius: 2, color: '#4b5563', borderColor: '#4b5563', fontSize: '0.75rem' }}
                                                        >
                                                            MD
                                                        </Button>
                                                    </Tooltip>
                                                    <Tooltip title="View Raw Report JSON">
                                                        <Button variant="outlined" onClick={() => setRawModalOpen(true)} startIcon={<CodeIcon />} size="small" sx={{ textTransform: 'none', borderRadius: 2, color: '#64748b', borderColor: '#cbd5e1', fontSize: '0.75rem' }}>View Raw</Button>
                                                    </Tooltip>
                                                    <Button variant="contained" onClick={() => setActiveStep(0)} size="small" sx={{ textTransform: 'none', borderRadius: 2, bgcolor: '#6366f1' }}>Done</Button>
                                                </Box>
                                            </Box>

                                            {reportTab === 0 && (
                                                <Box sx={{ py: 3 }}>
                                                    <Grid container spacing={4}>
                                                        <Grid size={{ xs: 12, sm: 6, md: 6 }}>
                                                            <Typography sx={{ fontWeight: 700, mb: 3 }}>Policy Compliance Breakdown</Typography>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
                                                                {Object.entries(auditResults.results?.by_policy || {}).map(([policy, data]: [string, any]) => (
                                                                    <Box key={policy}>
                                                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                                                            <Typography sx={{ fontSize: '0.85rem', fontWeight: 600, color: '#374151' }}>{policy}</Typography>
                                                                            <Typography sx={{ fontSize: '0.85rem', fontWeight: 700, color: '#6366f1' }}>{(data.score * 100).toFixed(0)}%</Typography>
                                                                        </Box>
                                                                        <LinearProgress variant="determinate" value={data.score * 100} sx={{ height: 6, borderRadius: 3, bgcolor: '#f1f5f9', '& .MuiLinearProgress-bar': { bgcolor: data.score > 0.8 ? '#10b981' : (data.score > 0.5 ? '#f59e0b' : '#ef4444') } }} />
                                                                    </Box>
                                                                ))}
                                                            </Box>
                                                        </Grid>
                                                        <Grid size={{ xs: 12, sm: 6, md: 6 }}>
                                                            <Typography sx={{ fontWeight: 700, mb: 3 }}>Strategic Recommendations</Typography>
                                                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                                                {/* {(auditResults.recommendations || ['Review output sanitization rules', 'Update bias mitigation datasets', 'Enable real-time logging for policy violations']).map((rec: string, i: number) => ( */}
                                                                {auditResults.recommendations.map((rec: Recommendation, i: number) => (
                                                                    <Box key={i} sx={{ p: 2, bgcolor: alpha('#6366f1', 0.03), borderRadius: 2, border: '1px solid', borderColor: alpha('#6366f1', 0.1), display: 'flex', gap: 2 }}>
                                                                        <CheckCircleOutlinedIcon sx={{ color: '#10b981', fontSize: 20, mt: 0.3 }} />
                                                                        <Box>
                                                                            <Typography sx={{ fontSize: '0.9rem', fontWeight: 600, color: '#1e293b' }}>{rec.title}</Typography>
                                                                            <Typography sx={{ fontSize: '0.85rem', color: '#4b5563', mt: 0.5, lineHeight: 1.5 }}>{rec.description}</Typography>
                                                                            <Box sx={{ display: 'flex', gap: 1, mt: 1.5 }}>
                                                                                <Chip label={`Priority: ${rec.priority}`} size="small" sx={{ height: 20, fontSize: '0.65rem', bgcolor: rec.priority === 'high' ? '#fee2e2' : '#f1f5f9', color: rec.priority === 'high' ? '#991b1b' : '#64748b' }} />
                                                                                <Chip label={`Finding ID: ${rec.finding_id}`} size="small" sx={{ height: 20, fontSize: '0.65rem', bgcolor: '#f1f5f9' }} />
                                                                            </Box>
                                                                        </Box>
                                                                    </Box>
                                                                ))}
                                                            </Box>
                                                        </Grid>
                                                    </Grid>
                                                </Box>
                                            )}

                                            {reportTab === 1 && (
                                                <Box sx={{ py: 3 }}>
                                                    {Object.entries(auditResults.results?.framework_results || {}).map(([fw, data]: [string, any]) => (
                                                        <Box key={fw} sx={{ mb: 4 }}>
                                                            <Typography sx={{ fontWeight: 800, mb: 2, color: '#6366f1', fontSize: '0.85rem', textTransform: 'uppercase' }}>{fw.replace(/_/g, ' ')}</Typography>
                                                            <TableContainer sx={{ border: '1px solid #E8E8E8', borderRadius: 2 }}>
                                                                <Table size="small">
                                                                    <TableHead sx={{ bgcolor: '#f8fafc' }}>
                                                                        <TableRow>
                                                                            <TableCell sx={{ fontWeight: 700 }}>Test Case</TableCell>
                                                                            <TableCell sx={{ fontWeight: 700 }}>Category</TableCell>
                                                                            <TableCell sx={{ fontWeight: 700 }}>Result</TableCell>
                                                                            <TableCell sx={{ fontWeight: 700 }}>Explore</TableCell>
                                                                        </TableRow>
                                                                    </TableHead>
                                                                    <TableBody>
                                                                        {(data.tests || []).slice(0, 10).map((test: any, i: number) => (
                                                                            <TableRow key={i}>
                                                                                <TableCell sx={{ fontSize: '0.8rem' }}>{test.name || test.test_id}</TableCell>
                                                                                <TableCell><Chip label={test.category || 'General'} size="small" sx={{ height: 18, fontSize: '0.65rem' }} /></TableCell>
                                                                                <TableCell>
                                                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                                                                        {test.passed ? <CheckCircleOutlinedIcon sx={{ color: '#10b981', fontSize: 16 }} /> : <ErrorOutlineOutlinedIcon sx={{ color: '#ef4444', fontSize: 16 }} />}
                                                                                        <Typography sx={{ fontSize: '0.8rem', fontWeight: 600, color: test.passed ? '#10b981' : '#ef4444' }}>{test.passed ? 'PASS' : 'FAIL'}</Typography>
                                                                                    </Box>
                                                                                </TableCell>
                                                                                <TableCell><Button size="small" sx={{ minWidth: 0, p: 0.5 }}><InfoOutlinedIcon sx={{ fontSize: 16, color: '#6366f1' }} /></Button></TableCell>
                                                                            </TableRow>
                                                                        ))}
                                                                    </TableBody>
                                                                </Table>
                                                            </TableContainer>
                                                        </Box>
                                                    ))}
                                                </Box>
                                            )}

                                            {reportTab === 2 && (
                                                <Box sx={{ py: 3 }}>
                                                    <Grid container spacing={2}>
                                                        {(auditResults.findings || [
                                                            { severity: 'High', title: 'Adversarial vulnerability', description: 'Model is susceptible to specific prompt injection patterns.' },
                                                            { severity: 'Medium', title: 'Data leak', description: 'Possible leakage of system prompt information in edge cases.' }
                                                        ]).map((finding: any, i: number) => (
                                                            <Grid size={{ xs: 12, sm: 12, md: 12 }} key={i}>
                                                                <Box sx={{
                                                                    p: 2.5, borderRadius: 3, display: 'flex', gap: 2.5,
                                                                    border: '1px solid', borderColor: finding.severity === 'High' ? '#fed7aa' : '#fde68a',
                                                                    bgcolor: finding.severity === 'High' ? '#fffaf5' : '#fffdf0'
                                                                }}>
                                                                    <Box sx={{ px: 1.5, py: 0.4, borderRadius: 1.5, bgcolor: finding.severity === 'High' ? '#f97316' : '#f59e0b', color: '#fff', fontSize: '0.7rem', fontWeight: 800 }}>{finding.severity}</Box>
                                                                    <Box>
                                                                        <Typography sx={{ fontWeight: 700, mb: 0.5 }}>{finding.title}</Typography>
                                                                        <Typography sx={{ fontSize: '0.85rem', color: '#6b7280' }}>{finding.description}</Typography>
                                                                    </Box>
                                                                </Box>
                                                            </Grid>
                                                        ))}
                                                    </Grid>
                                                </Box>
                                            )}

                                            {reportTab === 3 && (
                                                <Box sx={{ py: 3 }}>
                                                    <Paper elevation={0} sx={{ bgcolor: '#1e293b', p: 3, borderRadius: 3, maxHeight: '60vh', overflow: 'auto' }}>
                                                        <pre style={{ margin: 0, color: '#94a3b8', fontSize: '0.8rem', fontFamily: 'monospace' }}>
                                                            {JSON.stringify(auditResults, null, 2)}
                                                        </pre>
                                                    </Paper>
                                                </Box>
                                            )}
                                        </Box>
                                    )}

                                    {/* Audit Logs - Only show when pending */}
                                    {!auditResults && (
                                        <Box sx={{ mt: 4 }}>
                                            <Typography sx={{ fontWeight: 600, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                                                <InfoOutlinedIcon sx={{ fontSize: 20, color: '#64748b' }} />
                                                Live Audit Logs
                                            </Typography>
                                            <Box sx={{ bgcolor: '#1e293b', p: 3, borderRadius: 3, maxHeight: 200, overflowY: 'auto' }}>
                                                {auditLogs.map((log, i) => (
                                                    <Typography key={i} sx={{ color: '#94a3b8', mb: 0.5, fontSize: '0.8rem', fontFamily: 'monospace' }}>
                                                        <span style={{ color: '#10b981' }}>[{new Date().toLocaleTimeString()}]</span> {log}
                                                    </Typography>
                                                ))}
                                                {running && <Typography sx={{ color: '#6366f1', mt: 1, animation: 'pulse 1.5s infinite' }}>_</Typography>}
                                            </Box>
                                        </Box>
                                    )}
                                </Box>
                            )}
                        </Paper>
                    </Box>
                </Box>
            )}

            {activeTab === 1 && (
                <Box sx={{ py: 12, textAlign: 'center', bgcolor: '#fff', borderRadius: 3, border: '1px solid #f1f5f9' }}>
                    <Typography color="text.secondary">Your audit history will be listed here.</Typography>
                </Box>
            )}

            {/* Raw Data Modal */}
            <Dialog
                open={rawModalOpen}
                onClose={() => setRawModalOpen(false)}
                maxWidth="md"
                fullWidth
                PaperProps={{
                    sx: { borderRadius: 3, bgcolor: '#ffffff' }
                }}
            >
                <DialogTitle sx={{ m: 0, p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #f1f5f9' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <CodeIcon sx={{ color: '#6366f1' }} />
                        <Typography sx={{ fontWeight: 700 }}>Raw Audit Report Data</Typography>
                    </Box>
                    <IconButton onClick={() => setRawModalOpen(false)} size="small" sx={{ color: '#94a3b8' }}>
                        <CloseIcon />
                    </IconButton>
                </DialogTitle>
                <DialogContent sx={{ p: 0 }}>
                    <Box sx={{ bgcolor: '#1e293b', p: 3, maxHeight: '70vh', overflow: 'auto' }}>
                        <pre style={{ margin: 0, color: '#94a3b8', fontSize: '0.8rem', fontFamily: 'monospace' }}>
                            {JSON.stringify(auditResults, null, 2)}
                        </pre>
                    </Box>
                </DialogContent>
                <DialogActions sx={{ p: 2, borderTop: '1px solid #f1f5f9' }}>
                    <Button onClick={() => setRawModalOpen(false)} sx={{ textTransform: 'none', fontWeight: 600, color: '#64748b' }}>Close</Button>
                    <Button
                        variant="contained"
                        onClick={() => {
                            const blob = new Blob([JSON.stringify(auditResults, null, 2)], { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `audit-report-${auditResults?.audit_id || 'raw'}.json`;
                            a.click();
                        }}
                        sx={{ textTransform: 'none', bgcolor: '#6366f1', borderRadius: 2 }}
                    >
                        Download JSON
                    </Button>
                </DialogActions>
            </Dialog>
        </Box>
    );
};

export default Audits;
