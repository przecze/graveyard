import { createRoot } from 'react-dom/client';
import VisitorView from './VisitorView';
import PolarWalker from './PolarWalker';
import './index.css';

const root = document.getElementById('root')!;
const app = window.location.pathname.startsWith('/v2')
  ? <PolarWalker />
  : <VisitorView />;

createRoot(root).render(app);
