import os
import glob
import pandas as pd
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

# 1. Aapka exact folder path jahan 45 sub-segment CSV files pari hain
folder_path = r"C:\Users\Saad\Downloads\Project\Data segmentation\Withdrawal_Segmentation\Sub_Segmentation"

# Folder ke andar mojood tamam .csv files dhoondhein
all_csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
print(f"Total sub-segment CSV files found: {len(all_csv_files)}")

# 2. Embedding Model Load Karein (384-dimensional)
print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Docker PostgreSQL Connection (Port 5433)
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "withdrawal_support_db"

connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)

# Safety check: Table ko fresh rakhne ke liye truncate karein
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE support_embeddings RESTART IDENTITY;"))

# 4. Har CSV file ko loop karke database mein insert karein
total_rows = 0
for file_path in all_csv_files:
    # File ke naam se sub-segment category nikalen (e.g., 'time_delay_duration_issues')
    category_name = os.path.splitext(os.path.basename(file_path))[0]
    df = pd.read_csv(file_path)
    
    if df.empty:
        continue

    # Columns ko safely handle karein
    subj_col = 'subject' if 'subject' in df.columns else df.columns[0]
    status_col = 'status' if 'status' in df.columns else None
    ticket_col = 'ticket_number' if 'ticket_number' in df.columns else None

    # Embedding ke liye text tayar karein (Sub-segment + Subject)
    combined_texts = (
        "Sub-segment: " + category_name.replace("_", " ") + 
        " | Issue: " + df[subj_col].fillna('').astype(str)
    ).tolist()

    embeddings = model.encode(combined_texts, show_progress_bar=False)

    # Response/Details column tayar karein
    details = []
    for idx, row in df.iterrows():
        t_num = f"Ticket #{row[ticket_col]}" if ticket_col else "Ticket"
        st = f"Status: {row[status_col]}" if status_col else ""
        details.append(f"{t_num} | {st}".strip(" |"))

    df_to_db = pd.DataFrame({
        'category_name': [category_name] * len(df),
        'ticket_subject': df[subj_col].fillna('').astype(str),
        'response_text': details,
        'embedding': [str(emb.tolist()) for emb in embeddings]
    })

    df_to_db.to_sql('support_embeddings', con=engine, if_exists='append', index=False, chunksize=500)
    total_rows += len(df)
    print(f"Ingested: {category_name} ({len(df)} rows)")

print(f"\nMukammal! Tamam {len(all_csv_files)} sub-segments ({total_rows} total rows) database mein save ho chuke hain!")