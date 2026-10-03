export const W=1280,H=720,DURATION=20;
export const colors={search:'#e2bb37',aggregate:'#ddd0ab',answer:'#202020'};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
const ease=x=>{x=Math.max(0,Math.min(1,x));return x*x*(3-2*x);};
const mix=(a,b,u)=>a+(b-a)*u;
const txt=(x,y,s,size=16,fill='#edf1f8',extra='')=>'<text x="'+x+'" y="'+y+'" font-size="'+size+'" fill="'+fill+'" '+extra+'>'+esc(s)+'</text>';
export function lines(s,limit=26,max=3){const out=[];let current='';for(const word of String(s||'').replace(/\s+/g,' ').split(' ').flatMap(w=>w.length>limit?w.match(new RegExp('.{1,'+limit+'}','g')):[w])){if((current+' '+word).trim().length>limit&&current){out.push(current);current=word;}else current+=(current?' ':'')+word;}if(current)out.push(current);if(out.length>max){out.length=max;out[max-1]=out[max-1].replace(/[.,;:]$/,'')+'…';}return out;}
export function stageAt(t){return t<4.8?'Planner · initial graph':t<7.5?'Jina · search':t<10?'Planner · revision':t<12?'Serper · fetch':t<16.7?'Answerer · branches':'Answerer · final report';}
export function layout(graph){
 const depths=Object.fromEntries(graph.nodes.map(n=>[n.id,0]));
 for(let p=0;p<graph.nodes.length;p++)for(const e of graph.edges)depths[e.target]=Math.max(depths[e.target]||0,(depths[e.source]||0)+1);
 const cols=[...new Set(Object.values(depths))].sort((a,b)=>a-b),positions={},width=Math.min(264,1040/cols.length-22);
 cols.forEach((col,ci)=>{const group=graph.nodes.filter(n=>depths[n.id]===col),step=Math.min(112,400/group.length);group.forEach((n,i)=>positions[n.id]={x:66+ci*(1148-width)/Math.max(1,cols.length-1),y:253+(400-step*group.length)/2+i*step,w:width,h:Math.min(92,step-9),depth:ci,index:i});});return positions;
}
function focus(pos,ids,targetX=430,maxZoom=1.65){
 const ps=ids.map(id=>pos[id]).filter(Boolean);if(!ps.length)return {z:1,x:0,y:0};
 const left=Math.min(...ps.map(p=>p.x)),right=Math.max(...ps.map(p=>p.x+p.w)),top=Math.min(...ps.map(p=>p.y)),bottom=Math.max(...ps.map(p=>p.y+p.h));
 const z=Math.min(maxZoom,370/(bottom-top+36),700/(right-left+40));return {z,x:targetX-z*(left+right)/2,y:445-z*(top+bottom)/2};
}
export function cameraAt(t,graph,pos){
 const full={z:1,x:0,y:0},search=graph.nodes.filter(n=>n.type==='search'),agg=graph.nodes.filter(n=>n.type==='aggregate'),ans=graph.nodes.filter(n=>n.type==='answer');
 const first=focus(pos,search.slice(0,3).map(n=>n.id),370,1.8),branches=focus(pos,agg.slice(0,3).map(n=>n.id),610,1.6),last=focus(pos,ans.map(n=>n.id),285,1.45);
 const keys=[[0,full],[3.8,full],[5.1,first],[6.8,first],[7.45,full],[9.4,full],[10.5,first],[11.5,first],[12.4,branches],[14.2,branches],[15.5,full],[16.4,full],[17.5,last],[20,last]];
 let i=0;while(i<keys.length-2&&t>keys[i+1][0])i++;const [a,A]=keys[i],[b,B]=keys[i+1],u=ease((t-a)/(b-a));return {z:mix(A.z,B.z,u),x:mix(A.x,B.x,u),y:mix(A.y,B.y,u)};
}
function dots(x,y,t,c){return [0,1,2].map(i=>'<circle cx="'+(x+i*7)+'" cy="'+(y-Math.max(0,Math.sin(t*7-i*.8))*2)+'" r="1.7" fill="'+c+'" opacity="'+(.3+.7*Math.max(0,Math.sin(t*7-i*.8)))+'"/>').join('');}
const signature=n=>JSON.stringify([n.type,n.need,n.query||'']);
export function revealTimes(d,graph,pos,revision){
 const order=[...graph.nodes].sort((a,b)=>pos[a.id].depth-pos[b.id].depth||pos[a.id].index-pos[b.id].index);
 const initial=new Set(d.initial_graph.nodes.map(signature));
 const added=order.filter(n=>!initial.has(signature(n)));
 return Object.fromEntries(order.map((n,i)=>[n.id,revision?(initial.has(signature(n))?-1:7.55+added.indexOf(n)*1.7/Math.max(1,added.length-1)):.08+i*3.9/Math.max(1,order.length-1)]));
}
export function scene(d,t,{selected='',overview=false,video=false,theme='paper'}={}){
 t=Math.max(0,Math.min(20,t));
 const revision=t>=7.5,graph=revision?d.graph:d.initial_graph,pos=layout(graph);
 const cam=overview?{z:1,x:0,y:0}:cameraAt(t,graph,pos),birth=revealTimes(d,graph,pos,revision);
 const aggregates=graph.nodes.filter(n=>n.type==='aggregate'),depths=[...new Set(aggregates.map(n=>pos[n.id].depth))].sort((a,b)=>a-b),sources=Object.entries(d.citations||{});
 const visible=id=>overview?1:ease((t-birth[id])/.34);
 const white='#ffffff',ink='#202020',muted='#6b6b6b',border='#d9d9d9';
 let svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720" role="group" aria-label="'+esc(stageAt(t))+'" style="font-family:Geist,Arial,sans-serif"><defs><clipPath id="viewport"><rect x="28" y="249" width="1224" height="400"/></clipPath><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0 0 L5 2.5 L0 5" fill="#777777"/></marker></defs><rect width="1280" height="720" fill="'+white+'"/>';
 // The question is the only card heading; the page already carries the brand.
 const queryLines=lines(d.query,92,6),qs=queryLines.length>4?23:26,qstep=queryLines.length>4?28:33;
 queryLines.forEach((s,i)=>svg+=txt(46,49+i*qstep,s,qs,ink,'font-weight="550"'));
 svg+='<path d="M46 209 H1234" stroke="'+border+'"/>'+txt(47,236,overview?'Executed graph':stageAt(t),16,ink);
 if(t<4.8&&!overview)svg+=dots(246,232,t,ink);
 svg+='<g clip-path="url(#viewport)"><g transform="translate('+cam.x+' '+cam.y+') scale('+cam.z+')">';
 for(const [ei,e] of graph.edges.entries()){
  const a=pos[e.source],b=pos[e.target];if(!a||!b)continue;
  const progress=overview?1:ease((t-Math.max(birth[e.source],birth[e.target])-.2)/.45);
  if(!progress||!visible(e.source)||!visible(e.target))continue;
  const x1=a.x+a.w,y1=a.y+a.h/2,x2=b.x-4,y2=b.y+b.h/2,c=(x2-x1)*.5,path='M'+x1+' '+y1+' C'+(x1+c)+' '+y1+','+(x2-c)+' '+y2+','+x2+' '+y2;
  let length=0,previous={x:x1,y:y1};
  for(let step=1;step<=24;step++){const u=step/24,v=1-u,next={x:v*v*v*x1+3*v*v*u*(x1+c)+3*v*u*u*(x2-c)+u*u*u*x2,y:v*v*v*y1+3*v*v*u*y1+3*v*u*u*y2+u*u*u*y2};length+=Math.hypot(next.x-previous.x,next.y-previous.y);previous=next;}
  svg+='<path data-edge="'+esc(e.source+'-'+e.target)+'" d="'+path+'" stroke-dasharray="'+length+'" stroke-dashoffset="'+((1-progress)*length)+'" fill="none" stroke="#777777" stroke-width="1.4" opacity=".55" '+(progress>.99?'marker-end="url(#arrow)"':'')+'/>';
  if(t>=12&&progress>.99&&!overview){const u=((t-12)*.5+ei*.127)%1,v=1-u,x=v*v*v*x1+3*v*v*u*(x1+c)+3*v*u*u*(x2-c)+u*u*u*x2,y=v*v*v*y1+3*v*v*u*y1+3*v*u*u*y2+u*u*u*y2;svg+='<circle cx="'+x+'" cy="'+y+'" r="2.6" fill="#b58b18"/>';}
 }
 for(const n of graph.nodes){
  const reveal=visible(n.id);if(!reveal)continue;
  const p=pos[n.id],wave=depths.indexOf(p.depth);
  const start=n.type==='search'?Math.max(birth[n.id]+.4,revision&&birth[n.id]>0?10:4.8)+p.index*.08:n.type==='aggregate'?12+wave*4.4/Math.max(1,depths.length):16.7;
  const end=n.type==='search'?start+.6:n.type==='aggregate'?start+3.6/Math.max(1,depths.length):18.1;
  const generating=!overview&&t>=start&&t<end,done=t>=end,sy=p.y+(1-reveal)*8,answer=n.type==='answer';
  const fill=answer?ink:n.type==='search'?'#f4d86a':'#f5efde',fg=answer?white:ink,sub=answer?'#dddddd':'#555047',label=n.type==='aggregate'?'BRANCH':n.type.toUpperCase();
  svg+='<g data-node="'+esc(n.id)+'" role="button" tabindex="0" aria-label="'+esc(n.id+': '+n.need)+'" opacity="'+reveal+'" class="graph-node" style="cursor:pointer"><title>'+esc(n.need)+'</title><rect x="'+p.x+'" y="'+sy+'" width="'+p.w+'" height="'+p.h+'" rx="5" fill="'+fill+'" stroke="'+(selected===n.id||generating?ink:answer?ink:'#cbbf9e')+'" stroke-width="'+(selected===n.id?2.5:generating?1.8:1)+'"/>';
  svg+=txt(p.x+12,sy+18,n.id,11,fg,'font-weight="650"')+txt(p.x+p.w-11,sy+18,label,10,sub,'text-anchor="end"');
  lines(n.need,Math.floor((p.w-24)/6.2),p.h<67?1:2).forEach((s,i)=>svg+=txt(p.x+12,sy+38+i*16,s,12,fg));
  if(p.h>76){const status=generating?(n.type==='search'?'Searching':'Writing'):done?(n.type==='search'?'Retrieved':'Complete'):'Planned';svg+=txt(p.x+12,sy+p.h-9,status,10,sub);if(generating)svg+=dots(p.x+78,sy+p.h-11,t,fg);if(done)svg+='<path d="M'+(p.x+p.w-22)+' '+(sy+p.h-14)+' l3 3 l6 -7" fill="none" stroke="'+fg+'" stroke-width="1.4"/>';}
  else if(generating)svg+=dots(p.x+p.w-29,sy+17,t,fg);
  svg+='</g>';
 }
 svg+='</g></g>';
 const retrieval=t>=4.8&&t<7.4||t>=10&&t<12.1;
 if(retrieval&&!overview){
  const fetch=t>=10,elapsed=t-(fetch?10:4.8),a=ease(elapsed/.35)*(1-ease((t-(fetch?11.7:7.05))/.35));
  const items=fetch?sources.filter(([,s])=>s.kind==='fetch').slice(0,3):graph.nodes.filter(n=>n.type==='search').slice(0,3).map(n=>[n.id,{title:n.query||n.need}]);
  const accent=fetch?'#202020':'#b58b18',surface=fetch?'#f5efde':'#fff5c7';
  svg+='<g opacity="'+a+'"><rect x="735" y="296" width="465" height="286" rx="6" fill="'+surface+'" stroke="#d0c6a9"/>'+txt(762,330,fetch?'Serper · page retrieval':'Jina · search',20,ink,'font-weight="550"');
  if(fetch&&d.retrieval_status){const s=d.retrieval_status;svg+=txt(762,560,s.succeeded+' / '+s.requested+' pages returned'+(s.succeeded<s.requested?' · unavailable pages skipped':''),12,muted);}
  for(let i=0;i<items.length;i++){
   const yy=353+i*58,ready=elapsed>.5+i*.35,title=items[i][1].title;
   svg+='<rect x="760" y="'+yy+'" width="413" height="47" rx="3" fill="#ffffff" stroke="#e0d9c5"/>'+txt(774,yy+19,lines(title,49,1)[0],12,ink);
   if(ready){svg+='<path d="M775 '+(yy+33)+' h'+(fetch?240:190+i*36)+'" stroke="'+accent+'" stroke-width="2"/><path d="M1145 '+(yy+26)+' l4 4 l7 -9" fill="none" stroke="'+accent+'" stroke-width="1.6"/>';}
   else svg+=dots(775,yy+33,t,accent);
  }
  for(let i=0;i<4;i++){const u=(elapsed*.8+i*.25)%1;svg+='<rect x="'+(610+u*120)+'" y="'+(400+i*36)+'" width="'+(fetch?7:4)+'" height="'+(fetch?9:4)+'" rx="1" fill="'+accent+'" opacity="'+Math.sin(u*Math.PI)+'"/>';}svg+='</g>';
 }
 // A short close-up makes within-wave branch execution visible. These are
 // excerpts of saved outputs, not a claim that token streaming was recorded.
 if(t>=12.8&&t<15&&!overview){
  const wave=(d.execution_waves||[]).find(ids=>ids.length>1)||(d.execution_waves||[])[0]||[];
  const branches=wave.map(id=>graph.nodes.find(n=>n.id===id&&n.type==='aggregate')).filter(Boolean).slice(0,3);
  if(branches.length){
   const elapsed=t-12.8,alpha=ease(elapsed/.18)*(1-ease((t-14.8)/.2)),gap=20,width=(1152-gap*(branches.length-1))/branches.length;
   svg+='<g data-branch-writing="true" opacity="'+alpha+'"><rect x="28" y="249" width="1224" height="400" fill="#ffffff" fill-opacity=".97"/>'+txt(64,282,branches.length>1?'Branch reports · same execution wave':'Branch report',18,ink,'font-weight="550"');
   if(wave.length>3)svg+=txt(1216,282,'3 of '+wave.length+' branches',12,muted,'text-anchor="end"');
   branches.forEach((n,i)=>{
    const x=64+i*(width+gap),limit=Math.floor((width-36)/7.2),progress=Math.max(0,Math.min(1,(elapsed-.2-i*.04)/1.55));
    const report=String(d.outputs[n.id]||'').replace(/^#+\s+.*$/gm,'').replace(/[*_#]/g,'').replace(/\s+/g,' ').trim();
    const excerpt=lines(report,limit,6).join('\n'),typed=excerpt.slice(0,Math.floor(excerpt.length*progress));
    svg+='<g data-node="'+esc(n.id)+'" role="button" tabindex="0" aria-label="'+esc(n.id+': '+n.need)+'" class="graph-node" style="cursor:pointer"><rect x="'+x+'" y="302" width="'+width+'" height="315" rx="5" fill="#f5efde" stroke="#d0c6a9"/>'+txt(x+18,330,n.id,13,ink,'font-weight="650"');
    lines(n.need,Math.floor((width-36)/8),2).forEach((line,j)=>svg+=txt(x+18,360+j*20,line,15,ink,'font-weight="550"'));
    svg+='<path d="M'+(x+18)+' 398 H'+(x+width-18)+'" stroke="#d8cfb7"/>';
    typed.split('\n').forEach((line,j)=>svg+=txt(x+18,425+j*24,line,14,ink));
    svg+=txt(x+18,594,'Writing',12,muted)+dots(x+70,590,t+i*.3,ink)+'</g>';
   });
   svg+='</g>';
  }
 }
 if(t>17.35&&!overview){
  const a=ease((t-17.35)/.6),excerpt=String(d.answer).replace(/^#+\s+.*$/gm,'').replace(/[*_#]/g,'').trim();
  svg+='<g opacity="'+a+'"><rect x="512" y="284" width="692" height="323" rx="6" fill="#ffffff" stroke="#c8c8c8"/>'+txt(539,319,'Final report',18,ink,'font-weight="550"');
  lines(excerpt,72,8).forEach((s,i)=>svg+=txt(539,352+i*27,s,16,ink));svg+='</g>';
 }
 svg+='<rect x="46" y="674" width="1188" height="2" fill="#e5e5e5"/><rect x="46" y="674" width="'+1188*t/20+'" height="2" fill="#202020"/>'+txt(46,703,graph.nodes.filter(n=>n.type==='search').length+' searches · '+aggregates.length+' branches',12,muted)+txt(1234,703,overview?'Select a node':'Compressed replay',12,muted,'text-anchor="end"');
 return svg+'</svg>';
}
