import { createRoot } from 'react-dom/client';
import VisitorView from './VisitorView';
import PolarWalker from './PolarWalker';
import Walker from './graveyard/Walker';
import './index.css';

const root = document.getElementById('root')!;
const path = window.location.pathname;
const app = path.startsWith('/v2') ? <PolarWalker />
  : path.startsWith('/v1') ? <VisitorView />
  : <Walker />;

createRoot(root).render(app);
