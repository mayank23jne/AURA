import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

const client = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const api = {
    getStatus: () => client.get('/status'),
    getAudits: (params = {}) => client.get('/audits', { params }),
    getAgents: () => client.get('/agents'),
    getAgentMetrics: (name: string) => client.get(`/agents/${name}/metrics`),
    getModels: (params = {}) => client.get('/models', { params }),
    registerModel: (data: any) => client.post('/models', data),
    getPolicies: () => client.get('/policies'),
    generatePolicy: (data: any) => client.post('/policies/generate', data),
    createPolicy: (data: any) => client.post('/policies', data),
    updatePolicy: (policyId: string, data: any) => client.put(`/policies/${policyId}`, data),
    deletePolicy: (policyId: string) => client.delete(`/policies/${policyId}`),
    uploadPolicies: (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        return client.post('/policies/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
    },
    runAudit: (data: any) => client.post('/audit', data),
    getAuditPDF: (auditId: string) => client.get(`/audits/${auditId}/pdf`, { responseType: 'blob' }),
};

export default client;
