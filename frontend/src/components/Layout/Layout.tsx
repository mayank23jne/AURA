import React, { useState } from 'react';
import { Box, Toolbar } from '@mui/material';
import Sidebar from './Sidebar';
import Header from './Header';

interface LayoutProps {
    children: React.ReactNode;
}

const SIDEBAR_WIDTH = 240;
const COLLAPSED_SIDEBAR_WIDTH = 80;

const Layout: React.FC<LayoutProps> = ({ children }) => {
    const [collapsed, setCollapsed] = useState(false);

    const handleToggleSidebar = () => {
        setCollapsed(!collapsed);
    };

    const currentSidebarWidth = collapsed ? COLLAPSED_SIDEBAR_WIDTH : SIDEBAR_WIDTH;

    return (
        <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
            <Sidebar collapsed={collapsed} onToggle={handleToggleSidebar} drawerWidth={SIDEBAR_WIDTH} collapsedWidth={COLLAPSED_SIDEBAR_WIDTH} />
            <Box
                component="main"
                sx={{
                    flexGrow: 1,
                    pl: 2,
                    pt: 2,
                    pb: 2,
                    pr: 2,
                    width: { sm: `calc(100% - ${currentSidebarWidth}px)` },
                    transition: (theme) => theme.transitions.create(['width', 'margin'], {
                        easing: theme.transitions.easing.sharp,
                        duration: theme.transitions.duration.enteringScreen,
                    }),
                }}
            >
                <Header collapsed={collapsed} sidebarWidth={currentSidebarWidth} />
                <Toolbar /> {/* Spacer for fixed Header */}
                {children}
            </Box>
        </Box>
    );
};

export default Layout;
