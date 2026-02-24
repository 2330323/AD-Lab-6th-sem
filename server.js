require('dotenv').config();
const express = require('express');
const mysql = require('mysql2');
const session = require('express-session');
const bcrypt = require('bcryptjs');
const svgCaptcha = require('svg-captcha');

const app = express();

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.static('public'));
app.set('view engine', 'ejs');

// Session Config
app.use(session({
    secret: 'secret_key_change_this',
    resave: false,
    saveUninitialized: true
}));

// Database Connection
const db = mysql.createConnection({
    host: 'localhost',
    user: 'admin',      
    password: 'Omi9321', 
    database: 'user_portal'
});

db.connect((err) => {
    if (err) console.error('DB Error: ' + err.stack);
    else console.log('Connected to MariaDB...');
});

// --- ROUTES ---

app.get('/', (req, res) => res.redirect('/login'));

// CAPTCHA
app.get('/captcha', (req, res) => {
    const captcha = svgCaptcha.create();
    req.session.captcha = captcha.text;
    res.type('svg');
    res.status(200).send(captcha.data);
});

// LOGIN
app.get('/login', (req, res) => {
    res.render('login', { error: null });
});

app.post('/login', (req, res) => {
    const { username, password, captcha } = req.body;

    if (req.session.captcha !== captcha) {
        return res.render('login', { error: 'Invalid Captcha! Try again.' });
    }

    db.query('SELECT * FROM users WHERE username = ?', [username], async (err, results) => {
        if (err) throw err;
        if (results.length === 0 || !(await bcrypt.compare(password, results[0].password))) {
            return res.render('login', { error: 'Invalid Credentials' });
        }
        req.session.user = results[0];
        res.redirect('/dashboard');
    });
});

// REGISTER
app.get('/register', (req, res) => {
    res.render('register', { error: null });
});

app.post('/register', async (req, res) => {
    const { first_name, last_name, username, email, password, confirm_password } = req.body;

    if (password !== confirm_password) {
        return res.render('register', { error: 'Passwords do not match!' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const sql = 'INSERT INTO users (first_name, last_name, username, email, password) VALUES (?, ?, ?, ?, ?)';

    db.query(sql, [first_name, last_name, username, email, hashedPassword], (err) => {
        if (err) return res.render('register', { error: 'Username or Email already taken.' });
        res.redirect('/login');
    });
});

// DASHBOARD
app.get('/dashboard', (req, res) => {
    if (!req.session.user) return res.redirect('/login');
    
    db.query('SELECT * FROM grades WHERE user_id = ?', [req.session.user.id], (err, grades) => {
        if (err) throw err;
        res.render('dashboard', { user: req.session.user, grades: grades });
    });
});

// Update Profile
app.post('/update-profile', (req, res) => {
    if (!req.session.user) return res.redirect('/login');

    const { username, email } = req.body;
    const userId = req.session.user.id;

    const sql = 'UPDATE users SET username = ?, email = ? WHERE id = ?';
    db.query(sql, [username, email, userId], (err) => {
        if (err) {
            console.error(err);
            // In a real app, handle duplicate email errors here
        } else {
            // Update session data so the UI refreshes immediately
            req.session.user.username = username;
            req.session.user.email = email;
        }
        res.redirect('/dashboard');
    });
});

// Reset Password
app.post('/reset-password', async (req, res) => {
    if (!req.session.user) return res.redirect('/login');

    const { new_password } = req.body;
    const hashedPassword = await bcrypt.hash(new_password, 10);
    const userId = req.session.user.id;

    db.query('UPDATE users SET password = ? WHERE id = ?', [hashedPassword, userId], (err) => {
        if (err) console.error(err);
        res.redirect('/dashboard');
    });
});

// LOGOUT
app.get('/logout', (req, res) => {
    req.session.destroy();
    res.redirect('/login');
});

app.listen(3000, () => console.log('Server running on http://localhost:3000'));