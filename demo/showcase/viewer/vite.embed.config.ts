import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/postcss';
import {fileURLToPath} from 'node:url';

export default defineConfig({
 root:fileURLToPath(new URL('./embed',import.meta.url)),
 base:'./',
 publicDir:fileURLToPath(new URL('./public',import.meta.url)),
 resolve:{alias:{'@':fileURLToPath(new URL('./',import.meta.url))}},
 plugins:[react()],
 css:{postcss:{plugins:[tailwindcss()]}},
 build:{outDir:fileURLToPath(new URL('./embed-dist',import.meta.url)),emptyOutDir:true},
});
