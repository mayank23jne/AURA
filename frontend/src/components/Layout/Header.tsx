import React from 'react';
import {
    AppBar,
    Toolbar,
    Typography,
    Box,
    InputBase,
    IconButton,
    Badge,
    Avatar,
    Breadcrumbs,
    Link
} from '@mui/material';
import {
    Search as SearchIcon,
    NotificationsNone as NotificationsIcon,
    ChevronRight as ChevronRightIcon,
    Home as HomeIcon
} from '@mui/icons-material';
import { styled, alpha } from '@mui/material/styles';

const Search = styled('div')(({ theme }) => ({
    position: 'relative',
    borderRadius: 10,
    backgroundColor: alpha(theme.palette.common.white, 1),
    border: '1px solid #e2e8f0',
    marginLeft: 0,
    width: '100%',
    [theme.breakpoints.up('sm')]: {
        marginLeft: theme.spacing(1),
        width: 'auto',
    },
}));

const SearchIconWrapper = styled('div')(({ theme }) => ({
    padding: theme.spacing(0, 2),
    height: '100%',
    position: 'absolute',
    pointerEvents: 'none',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#94a3b8'
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
    color: 'inherit',
    width: '100%',
    '& .MuiInputBase-input': {
        padding: theme.spacing(1, 1, 1, 0),
        paddingLeft: `calc(1em + ${theme.spacing(4)})`,
        transition: theme.transitions.create('width'),
        [theme.breakpoints.up('sm')]: {
            width: '20ch',
            '&:focus': {
                width: '30ch',
            },
        },
    },
}));

interface HeaderProps {
    collapsed: boolean;
    sidebarWidth: number;
}

const Header: React.FC<HeaderProps> = ({ sidebarWidth }) => {
    return (
        <AppBar
            position="fixed"
            sx={{
                borderRadius: 0,
                width: { sm: `calc(100% - ${sidebarWidth}px)` },
                ml: { sm: `${sidebarWidth}px` },
                bgcolor: '#FFFFFF',
                boxShadow: 'none',
                borderBottom: '1px solid #E8E8E8',
                color: 'text.primary',
                transition: (theme) => theme.transitions.create(['width', 'margin'], {
                    easing: theme.transitions.easing.sharp,
                    duration: theme.transitions.duration.enteringScreen,
                }),
            }}
        >
            <Toolbar sx={{ justifyContent: 'space-between', px: { sm: 4 }, minHeight: 64 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Breadcrumbs
                        separator={<ChevronRightIcon fontSize="small" sx={{ color: '#94a3b8' }} />}
                        aria-label="breadcrumb"
                    >
                        <Link underline="hover" color="inherit" href="/" sx={{ display: 'flex', alignItems: 'center' }}>
                            <HomeIcon fontSize="small" sx={{ color: '#94a3b8' }} />
                        </Link>
                        <Typography sx={{ fontSize: '0.875rem', fontWeight: 300, color: '#666D73' }}>
                            Dashboard
                        </Typography>
                    </Breadcrumbs>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Search sx={{ bgcolor: '#f1f5f9', border: 'none', minWidth: 280 }}>
                        <SearchIconWrapper>
                            <SearchIcon fontSize="small" />
                        </SearchIconWrapper>
                        <StyledInputBase
                            placeholder="Search…"
                            inputProps={{ 'aria-label': 'search' }}
                            sx={{ fontSize: '0.875rem' }}
                        />
                    </Search>

                    <IconButton size="small" sx={{ ml: 1, color: '#64748b' }}>
                        <Badge color="error" variant="dot" overlap="circular">
                            <NotificationsIcon />
                        </Badge>
                    </IconButton>

                    <Avatar
                        sx={{
                            width: 32,
                            height: 32,
                            ml: 1,
                            cursor: 'pointer',
                            border: '1.5px solid #e2e8f0'
                        }}
                        src="https://mui.com/static/images/avatar/1.jpg"
                    />
                </Box>
            </Toolbar>
        </AppBar>
    );
};

export default Header;
