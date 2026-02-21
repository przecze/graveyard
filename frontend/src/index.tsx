import { createRoot } from 'react-dom/client';
import App from './App';
import VisitorView from './VisitorView';
import './index.css';

const path = window.location.pathname.replace(/\/+$/, '') || '/';
const RootComponent = path === '/visit' ? VisitorView : App;

createRoot(document.getElementById('root')!).render(<RootComponent />);
