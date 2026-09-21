import {cpSync} from 'node:fs';
cpSync(new URL('./public/',import.meta.url),new URL('./embed-dist/',import.meta.url),{recursive:true});
cpSync(new URL('./embed/fonts/OFL.txt',import.meta.url),new URL('./embed-dist/assets/Geist-OFL.txt',import.meta.url));
