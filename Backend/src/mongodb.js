const mongoose = require("mongoose");
const bcrypt = require("bcrypt");
const jwt = require("jsonwebtoken");

// Connect to MongoDB
mongoose.connect("mongodb://localhost:27017/LoginSignUpDB")
  .then(() => {
    console.log("MongoDB connected");
  })
  .catch((error) => {
    console.error("Failed to connect to MongoDB:", error);
  });

// Define the schema for the user collection
const userSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    unique: true // Ensure unique usernames
  },
  password: {
    type: String,
    required: true
  }
});

// Middleware to hash the password before saving it to the database
userSchema.pre("save", async function(next) {
  // Only hash the password if it has been modified or is new
  if (!this.isModified("password")) return next();

  try {
    const hashedPassword = await bcrypt.hash(this.password, 10); // Hash the password with 10 rounds of salt
    console.log("Hashed password:", hashedPassword); // Log the hashed password
    this.password = hashedPassword; // Replace the plain password with the hashed one
    next();
  } catch (error) {
    next(error);
  }
});

// Method to generate JWT token for user
userSchema.methods.generateAuthToken = function() {
  const secretKey = "your_private_key"; // Replace "your_private_key" with your actual private key
  const token = jwt.sign({ _id: this._id }, secretKey); // Sign the token with the secret key
  return token;
};

// Add static method to check if a user with the same name and password already exists
userSchema.statics.userExists = async function(name, password) {
  const existingUser = await this.findOne({ name });
  if (!existingUser) return false;
  const isValidPassword = await bcrypt.compare(password, existingUser.password);
  return isValidPassword;
};

// Create the user model
const User = mongoose.model("User", userSchema);

module.exports = User;
