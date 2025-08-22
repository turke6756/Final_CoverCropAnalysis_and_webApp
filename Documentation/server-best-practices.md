# Server Best Practices for PMTiles Web App

## Quick Start

Use Node.js with byte-range support for PMTiles:

```bash
node -e "
const http = require('http');
const fs = require('fs');
const path = require('path');

const server = http.createServer((req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Range');
  res.setHeader('Accept-Ranges', 'bytes');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }
  
  let filePath = '.' + req.url;
  if (filePath === './') filePath = './deck_gl_polygon_dashboard_time_agnostic_fixed_v7_mvt2_PATCHED_v2.html';
  
  const extname = path.extname(filePath);
  let contentType = 'text/html';
  switch (extname) {
    case '.js': contentType = 'text/javascript'; break;
    case '.css': contentType = 'text/css'; break;
    case '.json': contentType = 'application/json'; break;
    case '.png': contentType = 'image/png'; break;
    case '.jpg': contentType = 'image/jpg'; break;
    case '.pmtiles': contentType = 'application/octet-stream'; break;
  }
  
  fs.stat(filePath, (err, stats) => {
    if (err) {
      res.writeHead(404);
      res.end('File not found');
      return;
    }
    
    const range = req.headers.range;
    if (range && extname === '.pmtiles') {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stats.size - 1;
      const chunksize = (end - start) + 1;
      
      const stream = fs.createReadStream(filePath, {start, end});
      res.writeHead(206, {
        'Content-Range': \`bytes \${start}-\${end}/\${stats.size}\`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunksize,
        'Content-Type': contentType
      });
      stream.pipe(res);
    } else {
      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': stats.size
      });
      fs.createReadStream(filePath).pipe(res);
    }
  });
});

const PORT = 8000;
server.listen(PORT, () => console.log('Server with byte-range support running at http://localhost:' + PORT));
"
```

## Why This Works

1. **HTTP Byte-Range Support**: PMTiles requires servers to support partial content requests (HTTP 206 responses)
2. **CORS Headers**: Enables cross-origin requests for web APIs
3. **Content-Length**: Required header for proper file serving
4. **Accept-Ranges**: Tells clients the server supports byte-range requests

## Common Issues and Solutions

### Issue: "Server returned no content-length header"
**Cause**: Basic HTTP servers don't provide Content-Length or byte-range support
**Solution**: Use the Node.js server above with proper headers

### Issue: Polygons not loading
**Cause**: Python's `http.server` doesn't support byte-range requests for PMTiles
**Solution**: Always use Node.js for PMTiles applications

### Issue: CORS errors
**Cause**: Missing Access-Control headers
**Solution**: Include comprehensive CORS headers as shown above

## What NOT to Use

❌ **Python HTTP Server**: `python -m http.server`
- No byte-range support
- PMTiles will fail to load

❌ **Basic Node.js Server**: Without byte-range handling
- Will cause PMTiles loading errors

## Browser Troubleshooting

If polygons still don't appear:
1. Hard refresh: `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac)
2. Clear browser cache
3. Check browser console for errors
4. Verify all data files are present:
   - `orchards.pmtiles`
   - `manifest.json`
   - `2018.json` through `2023.json`
   - `streaks.json`

## File Structure
```
ProductionWebAppforCODEX/
├── deck_gl_polygon_dashboard_time_agnostic_fixed_v7_mvt2_PATCHED_v2.html
├── orchards.pmtiles
├── manifest.json
├── 2018.json
├── 2019.json
├── 2020.json
├── 2021.json
├── 2022.json
├── 2023.json
└── streaks.json
```