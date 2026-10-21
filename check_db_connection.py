from app.database import test_connection

result = test_connection()

print("Database connection successful!")
print("Test query result:", result)