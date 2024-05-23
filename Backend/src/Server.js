// server.js

const express = require('express');
const mongoose = require('mongoose');
const bodyParser = require('body-parser');

const app = express();

// Middleware
app.use(bodyParser.json());

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/contact_form', {
    useNewUrlParser: true,
    useUnifiedTopology: true,
});
const db = mongoose.connection;
db.on('error', console.error.bind(console, 'MongoDB connection error:'));
db.once('open', () => console.log('Connected to MongoDB'));

// Define a schema for the contact form data
const contactSchema = new mongoose.Schema({
    name: String,
    email: String,
    message: String,
});

// Define a model for the contact form data
const Contact = mongoose.model('Contact', contactSchema);

// Endpoint to handle POST requests to /api/contact
app.post('/api/contact', async (req, res) => {
    try {
        const { name, email, message } = req.body;
        const newContact = new Contact({ name, email, message });
        await newContact.save();
        res.status(201).send('Message sent successfully!');
    } catch (error) {
        console.error('Error saving message to database:', error);
        res.status(500).send('An error occurred while processing your request.');
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
