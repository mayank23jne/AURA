import React from 'react';
import {
    Box,
    Drawer,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    IconButton,
} from '@mui/material';
import {
    ChevronLeft as ChevronLeftIcon,
    Menu as MenuIcon
} from '@mui/icons-material';
import { NavLink, useLocation } from 'react-router-dom';

interface SidebarProps {
    collapsed: boolean;
    onToggle: () => void;
    drawerWidth: number;
    collapsedWidth: number;
}

const menuItems = [
    { text: 'Dashboard', icon: <img src="./homegrey.png" />, activeicon: <img src="./home-2.png" />, path: '/dashboard' },
    { text: 'Agent', icon: <img src="./agentic.png" />, activeicon: <img src="./agentactive.png" />, path: '/agent' }, // Image shows two Dashboards, the second has an AI icon
    { text: 'Model', icon: <img src="./flag-2.png" />, activeicon: <img src="./modelactive.png" />, path: '/model' },
    { text: 'Audits', icon: <img src="./clipboard-tick.png" />, activeicon: <img src="./auditactive.png" />, path: '/audits' },
    { text: 'Policy', icon: <img src="./book.png" />, activeicon: <img src="./policyactive.png" />, path: '/policy' },
    { text: 'Reports', icon: <img src="./reports.png"  />, activeicon: <img src="./homegrey.png" />, path: '/reports' },
    { text: 'Analytics', icon: <img src="./analytics.png"  />, activeicon: <img src="./homegrey.png" />, path: '/analytics' },
    { text: 'Comparison', icon: <img src="./comparison.png"  />, activeicon: <img src="./homegrey.png" />, path: '/comparison' },
    { text: 'Metrics', icon: <img src="./metrics.png"  />, activeicon: <img src="./homegrey.png" />, path: '/metrics' },
    { text: 'Settings', icon: <img src="./setting-2.png"  />, activeicon: <img src="./homegrey.png" />, path: '/settings' },
];

const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggle, drawerWidth, collapsedWidth }) => {
    const location = useLocation();
    const currentWidth = collapsed ? collapsedWidth : drawerWidth;

    return (
        <Drawer
            variant="permanent"
            sx={{
                width: currentWidth,
                flexShrink: 0,
                whiteSpace: 'nowrap',
                boxSizing: 'border-box',
                [`& .MuiDrawer-paper`]: {
                    width: currentWidth,
                    boxSizing: 'border-box',
                    borderRight: '1px solid #E8E8E8',
                    backgroundColor: '#FFFFFF',
                    borderRadius: 0,
                    transition: (theme) => theme.transitions.create('width', {
                        easing: theme.transitions.easing.sharp,
                        duration: theme.transitions.duration.enteringScreen,
                    }),
                    overflowX: 'hidden',
                },
            }}
        >
            <Box sx={{
                p: collapsed ? 2 : 3,
                display: 'flex',
                alignItems: 'center',
                justifyContent: collapsed ? 'center' : 'space-between',
                minHeight: 64
            }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {/* <LogoIcon color="primary" sx={{ fontSize: 32 }} /> */}
                    {!collapsed ? (<img src="./Aura_Logo.png" />) : (<img src="./Grouplogo.png" />)}
                </Box>
                {!collapsed && (
                    <IconButton onClick={onToggle} size="small" sx={{ color: '#64748b' }}>
                        <ChevronLeftIcon />
                    </IconButton>
                )}
            </Box>

            {collapsed && (
                <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <IconButton onClick={onToggle} size="small" sx={{ color: '#64748b' }}>
                        <MenuIcon />
                    </IconButton>
                </Box>
            )}

            <Box sx={{ overflow: 'auto', mt: collapsed ? 0 : 2 }}>
                <List sx={{ px: collapsed ? 1 : 2 }}>
                    {menuItems.map((item) => {
                        const isActive = location.pathname === item.path;
                        return (
                            <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
                                <ListItemButton
                                    component={NavLink}
                                    to={item.path}
                                    sx={{
                                        borderRadius: 0,
                                        justifyContent: collapsed ? 'center' : 'initial',
                                        px: 2.5,
                                        backgroundColor: isActive ? 'rgba(99, 102, 241, 0.08)' : 'transparent',
                                        color: isActive ? 'primary.main' : 'text.secondary',
                                        '&.active': {
                                            backgroundColor: 'rgba(99, 102, 241, 0.08)',
                                            color: 'primary.main',
                                        },
                                        '&:hover': {
                                            backgroundColor: 'rgba(99, 102, 241, 0.04)',
                                        },
                                    }}
                                >
                                    <ListItemIcon sx={{
                                        minWidth: 0,
                                        mr: collapsed ? 0 : 2,
                                        justifyContent: 'center',
                                        color: isActive ? 'primary.main' : 'inherit'
                                    }}>
                                        {isActive ? item.activeicon : item.icon}
                                    </ListItemIcon>
                                    {!collapsed && (
                                        <ListItemText
                                            primary={item.text}
                                            primaryTypographyProps={{
                                                fontSize: '0.875rem',
                                                fontWeight: isActive ? 600 : 500,
                                                noWrap: true
                                            }}
                                        />
                                    )}
                                </ListItemButton>
                            </ListItem>
                        );
                    })}
                </List>
            </Box>
        </Drawer>
    );
};

export default Sidebar;
