const express = require('express');
const path = require("path");
const collection = require("./mongodb");
const nodemailer = require('nodemailer');
const bodyParser = require("body-parser");
const { PythonShell } = require("python-shell");

const app = express();
const templatePath = path.join(__dirname, '../tempelates');
const reactPort = 3000; // Port on which React frontend is running

app.use(express.json());
app.set("view engine", "hbs");
app.set("views", templatePath);
app.use(express.urlencoded({ extended: false }));
app.use(bodyParser.urlencoded({ extended: false }));

// Define routes
app.get("/", (req, res) => {
    res.render("login");
});

app.get("/signup", (req, res) => {
    res.render("signup");
});

app.post("/signup", async (req, res) => {
    const data = {
        name: req.body.name,
        password: req.body.password,
        gst_in: req.body.gst_in, // Adding GST IN from the form
        email: req.body.email // Adding email from the form
    };

    try {
        await collection.insertMany(data);

        // Send email with secret code and GST IN
        sendEmail(req.body.name, req.body.email, req.body.gst_in);

        // Redirect to login page after successful signup
        res.redirect("/"); // Redirecting to the login page
    } catch (error) {
        console.error(error);
        res.send("Failed to sign up.");
    }
});

app.post("/login", async (req, res) => {
    try {
        const check = await collection.findOne({ name: req.body.name });
        if (check.password === req.body.password) {
            // Redirect to React frontend upon successful login
            res.redirect(`http://localhost:${reactPort}`);
        } else {
            res.send("Wrong password");
        }
    } catch (error) {
        console.error(error);
        res.send("Wrong details");
    }
});

// Sentiment Analysis route
app.get("/sentiment", (req, res) => {
    res.render("index");
});

app.post("/predict", (req, res) => {
    const message = req.body.message;

    // Configure python-shell options
    const options = {
        scriptPath: 'D:/MajorProject/Backend/data', // Specify the directory where your Python script is located
        args: [message] // Pass message as argument to the Python script
    };

    // Execute the Python script
    PythonShell.run('Sentiment_Analysis.py', options, (err, result) => {
        if (err) {
            console.error('Error:', err);
            res.status(500).send('Internal Server Error');
        } else {
            // Process the result
            const prediction = result[0]; // Assuming the Python script returns a single value (positive or negative)
            res.render("result", { prediction });
        }
    });
});

// Email sending function
function sendEmail(name, email, gst_in) {
    const transporter = nodemailer.createTransport({
        service: 'Gmail',
        auth: {
            user: 'vnagle454@gmail.com',
            pass: 'hghj zgnn sbmw jfnf'
        }
    });

    const mailOptions = {
        from: 'vnagle454@gmail.com',
        to: email,
        subject: 'Your Secret Code for Login',
        text: `Hello ${name},\n\nYour secret code for login is: sentiment&234\n\nGST IN: ${gst_in}`
    };

    transporter.sendMail(mailOptions, (error, info) => {
        if (error) {
            console.error('Error occurred while sending email:', error);
        } else {
            console.log('Email sent:', info.response);
        }
    });
}

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});
