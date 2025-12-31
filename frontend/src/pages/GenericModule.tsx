import React from 'react';
import { Box, Typography, Paper } from '@mui/material';

interface GenericModuleProps {
    title: string;
}

const GenericModule: React.FC<GenericModuleProps> = ({ title }) => {
    return (
        <Box sx={{ mt: 2 }}>
            <Typography variant="h4" sx={{ mb: 3, fontWeight: 700 }}>{title}</Typography>
            <Paper sx={{ p: 4, textAlign: 'center' }}>
                <Typography variant="h6" color="text.secondary">
                    {title} Module Implementation in Progress
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    This page is being developed based on the AURA Platform design specifications.
                </Typography>
            </Paper>
        </Box>
    );
};

export default GenericModule;
