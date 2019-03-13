var express = require('express');
var path = require('path');

// init express server
var app = express();
app.use(express.static(path.join(__dirname, '/dist')));
console.log(__dirname)

app.set('port', (process.env.PORT || 4200));

app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist/index.html'));
});

app.listen(app.get('port'),"0.0.0.0", function(){
  console.log("SERVER UP!")
})
