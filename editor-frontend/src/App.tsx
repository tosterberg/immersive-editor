import * as React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Link from '@mui/material/Link';
import ProTip from './ProTip';
import Article from "./Article";

function Copyright() {
  return (
    <Typography variant="body2" color="text.secondary" align="center">
      {'Copyright © '}
      <Link color="inherit" href="https://mui.com/">
        Your Website
      </Link>{' '}
      {new Date().getFullYear()}.
    </Typography>
  );
}

export default function App() {
  return (
    <Container maxWidth="md">
      <Box sx={{ my: 8 }}>
        <Typography variant="h4" component="h1" gutterBottom>
         Immersive Editor
        </Typography>
        <Article />
        <ProTip />
        <Copyright />
      </Box>
    </Container>
  );
}
