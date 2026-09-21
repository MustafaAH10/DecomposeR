import React from 'react';
import {createRoot} from 'react-dom/client';
import Home from '../app/page';
import '../app/globals.css';
import './embed.css';

createRoot(document.getElementById('root')!).render(<Home embedded assetBase="."/>);
// Only this embedded document reports its content height. No parent DOM access.
const reportSize=()=>{
 const main=document.querySelector('main');
 if(main&&window.parent!==window)window.parent.postMessage({type:'decomposer:height',height:Math.ceil(main.getBoundingClientRect().height)+12},window.location.origin);
};
const observer=new ResizeObserver(reportSize);
observer.observe(document.getElementById('root')!);
window.addEventListener('load',reportSize);
