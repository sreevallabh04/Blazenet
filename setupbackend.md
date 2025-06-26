# 🐘 PostgreSQL Backend Setup Guide for BlazeNet

This guide will walk you through setting up PostgreSQL with PostGIS for the BlazeNet system, step by step. No prior PostgreSQL knowledge required!

## 📋 What You'll Need

PostgreSQL is a powerful, open-source database that BlazeNet uses to store geospatial fire data, predictions, and simulation results. PostGIS is an extension that adds geospatial capabilities.

## 🚀 Step-by-Step Setup

### Step 1: Download PostgreSQL

1. **Open your web browser** and go to: https://www.postgresql.org/download/windows/
2. **Click** the "Download the installer" link
3. **Select** the latest version (PostgreSQL 15 or 16)
4. **Choose** "Windows x86-64" for 64-bit systems (most common)
5. **Click Download** - this will download a file like `postgresql-15.x-x-windows-x64.exe`

### Step 2: Install PostgreSQL

1. **Locate** the downloaded file (usually in your Downloads folder)
2. **Right-click** the installer and select "Run as administrator"
3. **Follow the installation wizard:**

#### Installation Wizard Steps:
- **Welcome Screen**: Click "Next"
- **Installation Directory**: Keep default (`C:\Program Files\PostgreSQL\15`) → Click "Next"
- **Select Components**: 
  - ✅ PostgreSQL Server
  - ✅ pgAdmin 4 (web-based admin tool)
  - ✅ Stack Builder (for extensions)
  - ✅ Command Line Tools
  - Click "Next"
- **Data Directory**: Keep default (`C:\Program Files\PostgreSQL\15\data`) → Click "Next"
- **Password**: 
  - Enter a password for the `postgres` superuser (REMEMBER THIS!)
  - Example: `blazenet2024` (write this down!)
  - Re-enter the same password
  - Click "Next"
- **Port**: Keep default `5432` → Click "Next"
- **Advanced Options**: Keep default locale → Click "Next"
- **Pre Installation Summary**: Review and click "Next"
- **Ready to Install**: Click "Next"
- **Installing**: Wait for completion (this may take several minutes)

### Step 3: Install PostGIS Extension

After PostgreSQL installs:

1. **Stack Builder** should automatically open
2. If not, find "Stack Builder" in your Start menu under "PostgreSQL 15"
3. **In Stack Builder:**
   - **Server**: Select "PostgreSQL 15 on port 5432"
   - Click "Next"
   - **Categories**: Expand "Spatial Extensions"
   - ✅ Check "PostGIS 3.x Bundle for PostgreSQL"
   - Click "Next"
   - **Download Directory**: Keep default → Click "Next"
   - **Download**: Click "Next" to download
   - **Installation**: Follow the PostGIS installer (keep all defaults)

### Step 4: Verify Installation

1. **Open Command Prompt** (Win + R, type `cmd`, press Enter)
2. **Test PostgreSQL** by typing:
   ```cmd
   psql --version
   ```
   You should see something like: `psql (PostgreSQL) 15.x`

### Step 5: Configure Database for BlazeNet

#### Option A: Using pgAdmin (Graphical Interface - Recommended for Beginners)

1. **Open pgAdmin 4**:
   - Find "pgAdmin 4" in your Start menu
   - It will open in your web browser
   - If prompted, set a master password (remember this too!)

2. **Connect to PostgreSQL**:
   - Click on "Servers" in the left panel
   - Right-click "PostgreSQL 15" and select "Connect Server"
   - Enter the password you set during installation
   - Click "Save password" and "OK"

3. **Create the BlazeNet Database**:
   - Right-click "Databases" under "PostgreSQL 15"
   - Select "Create" → "Database..."
   - **Database name**: `blazenet_db`
   - **Owner**: `postgres`
   - Click "Save"

4. **Enable PostGIS Extension**:
   - Click on the newly created `blazenet_db`
   - Right-click "Extensions"
   - Select "Create" → "Extension..."
   - **Name**: Select `postgis` from dropdown
   - Click "Save"

5. **Create BlazeNet User**:
   - Right-click "Login/Group Roles" under "PostgreSQL 15"
   - Select "Create" → "Login/Group Role..."
   - **General Tab**:
     - **Name**: `blazenet`
   - **Definition Tab**:
     - **Password**: `password` (or choose your own)
   - **Privileges Tab**:
     - ✅ Check "Can login?"
     - ✅ Check "Create databases?"
   - Click "Save"

6. **Grant Permissions**:
   - Right-click on `blazenet_db` → "Properties"
   - Go to "Security" tab
   - Click "+" to add a new privilege
   - **Grantee**: Select `blazenet`
   - **Privileges**: Check "ALL"
   - Click "Save"

#### Option B: Using Command Line (Advanced Users)

If you prefer command line:

1. **Open Command Prompt as Administrator**
2. **Connect to PostgreSQL**:
   ```cmd
   psql -U postgres -h localhost
   ```
   Enter your postgres password when prompted

3. **Run these commands** (press Enter after each line):
   ```sql
   CREATE DATABASE blazenet_db;
   CREATE USER blazenet WITH PASSWORD 'password';
   GRANT ALL PRIVILEGES ON DATABASE blazenet_db TO blazenet;
   \connect blazenet_db
   CREATE EXTENSION postgis;
   \quit
   ```

### Step 6: Update BlazeNet Configuration

1. **Open File Explorer** and navigate to your BlazeNet project folder:
   ```
   C:\Users\sriva\OneDrive\Desktop\Blazenet
   ```

2. **Open the file** `config.env` in any text editor (Notepad, VS Code, etc.)

3. **Update the database settings** (if they're not already correct):
   ```env
   # Database Configuration
   DATABASE_URL=postgresql://blazenet:password@localhost:5432/blazenet_db
   POSTGRES_USER=blazenet
   POSTGRES_PASSWORD=password
   POSTGRES_DB=blazenet_db
   ```

4. **Save the file**

### Step 7: Test the Connection

1. **Open Command Prompt** in your BlazeNet directory:
   ```cmd
   cd C:\Users\sriva\OneDrive\Desktop\Blazenet
   ```

2. **Test database connection**:
   ```cmd
   psql -U blazenet -h localhost -d blazenet_db
   ```
   - Enter password: `password`
   - If successful, you'll see: `blazenet_db=>`
   - Type `\quit` to exit

### Step 8: Start BlazeNet Services

Now you can start the BlazeNet system:

1. **In Command Prompt** (in your BlazeNet directory):
   ```cmd
   docker-compose up -d db
   ```
   Wait for it to start, then:
   ```cmd
   docker-compose up -d
   ```

2. **Verify everything is running**:
   ```cmd
   docker-compose ps
   ```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue: "psql: command not found"
**Solution**: Add PostgreSQL to your PATH:
1. Press Win + R, type `sysdm.cpl`, press Enter
2. Click "Environment Variables"
3. Under "System Variables", find and select "Path"
4. Click "Edit" → "New"
5. Add: `C:\Program Files\PostgreSQL\15\bin`
6. Click "OK" and restart Command Prompt

#### Issue: "Connection refused"
**Solution**: Make sure PostgreSQL service is running:
1. Press Win + R, type `services.msc`, press Enter
2. Find "postgresql-x64-15" in the list
3. Right-click → "Start" if it's not running

#### Issue: "Password authentication failed"
**Solution**: Reset the password:
1. Open pgAdmin 4
2. Right-click your server → "Properties"
3. Go to "Connection" tab and update the password

#### Issue: PostGIS extension not found
**Solution**: Reinstall PostGIS:
1. Run Stack Builder again
2. Select PostGIS bundle and reinstall

## 📋 Quick Reference

### Important Information to Remember:
- **PostgreSQL Port**: 5432
- **Database Name**: blazenet_db
- **Username**: blazenet
- **Password**: password (or what you set)
- **Superuser**: postgres
- **Superuser Password**: [what you set during installation]

### Useful Commands:
```bash
# Connect to database
psql -U blazenet -h localhost -d blazenet_db

# Check if PostGIS is installed
SELECT PostGIS_Version();

# List all databases
\l

# List all tables
\dt

# Exit psql
\quit
```

### File Locations:
- **PostgreSQL Installation**: `C:\Program Files\PostgreSQL\15\`
- **Data Directory**: `C:\Program Files\PostgreSQL\15\data\`
- **Config File**: `C:\Program Files\PostgreSQL\15\data\postgresql.conf`
- **BlazeNet Config**: `C:\Users\sriva\OneDrive\Desktop\Blazenet\config.env`

## ✅ Verification Checklist

Before proceeding, make sure:
- [ ] PostgreSQL 15+ is installed
- [ ] PostGIS extension is installed
- [ ] `blazenet_db` database exists
- [ ] `blazenet` user exists with correct permissions
- [ ] You can connect using: `psql -U blazenet -h localhost -d blazenet_db`
- [ ] PostGIS is enabled (run `SELECT PostGIS_Version();` in psql)
- [ ] `config.env` file has correct database settings

## 🎉 Next Steps

Once PostgreSQL is set up:
1. **Start BlazeNet**: `docker-compose up -d`
2. **Access the API**: http://localhost:8000/docs
3. **Access the Dashboard**: http://localhost:8501
4. **Check Health**: http://localhost:8000/health

## 🆘 Need Help?

If you encounter issues:
1. **Check the logs**: `docker-compose logs db`
2. **Restart services**: `docker-compose restart`
3. **Reset everything**: `docker-compose down -v` then `docker-compose up -d`

Remember: The most common issues are usually password-related or PostgreSQL service not running. Double-check these first!

---

**🎯 You're now ready to use BlazeNet with a fully configured PostgreSQL backend!** 