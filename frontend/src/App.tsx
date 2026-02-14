import { useState, useEffect } from 'react';
import './App.css';

const GRID_W = 21;
const GRID_H = 15;

export default function App() {
  const [pos, setPos] = useState({ x: Math.floor(GRID_W / 2), y: Math.floor(GRID_H / 2) });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      setPos(p => {
        let { x, y } = p;
        if (e.key === 'ArrowUp')    y = Math.max(0, y - 1);
        if (e.key === 'ArrowDown')  y = Math.min(GRID_H - 1, y + 1);
        if (e.key === 'ArrowLeft')  x = Math.max(0, x - 1);
        if (e.key === 'ArrowRight') x = Math.min(GRID_W - 1, x + 1);
        return { x, y };
      });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const rows: string[] = [];
  for (let y = 0; y < GRID_H; y++) {
    let row = '';
    for (let x = 0; x < GRID_W; x++) {
      row += (x === pos.x && y === pos.y) ? '@' : '+';
      if (x < GRID_W - 1) row += ' ';
    }
    rows.push(row);
  }

  return (
    <pre className="grid">
      {rows.join('\n')}
    </pre>
  );
}
