import psycopg2
from sentence_transformers import SentenceTransformer
import ollama

print("Loading local embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

LOCAL_LLM_MODEL = "llama3.2"
CONFIDENCE_THRESHOLD = 0.40

def retrieve_similar_tickets(user_query, top_k=5):
    query_embedding = embed_model.encode(user_query).tolist()
    
    conn = psycopg2.connect(
        dbname="withdrawal_support_db",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5433"
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT category_name, ticket_subject, response_text, 
               1 - (embedding <=> %s::vector) AS similarity
        FROM support_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """, (str(query_embedding), str(query_embedding), top_k))
    
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def get_category_instruction(category):
    cat = category.lower()
    if "kyc" in cat or "unverified" in cat:
        return (
            "Inform the customer that withdrawals require approved KYC verification. "
            "Ask ONLY for: (1) Registered Email/User ID, (2) Whether ID & Selfie documents have been submitted, "
            "and (3) Any rejection reason shown on their dashboard."
        )
    elif "pending" in cat or "delay" in cat or "time_" in cat or "release" in cat:
        return (
            "Acknowledge the withdrawal delay empathetically and explain that requests undergo standard batch/security processing. "
            "Ask ONLY for: (1) Ticket ID, (2) Withdrawal Request Date & Amount, and (3) Payment Method."
        )
    elif "crypto" in cat or "mcoin" in cat or "riser" in cat or "wallet" in cat or "staking" in cat:
        return (
            "Acknowledge the crypto/wallet transaction issue empathetically. "
            "Ask ONLY for: (1) Transaction Hash (TxID), (2) Wallet Address, (3) Network used (e.g., TRC20/ERC20/BEP20), and (4) Transaction Amount."
        )
    elif "bank" in cat or "ebank" in cat or "mcash" in cat or "mcard" in cat:
        return (
            "Explain that bank/e-bank transfers can take extra clearing time depending on the banking partner. "
            "Ask ONLY for: (1) Withdrawal Reference ID, (2) Bank/E-Bank Name, (3) Account Title & last 4 digits, and (4) Bank statement screenshot if marked Completed."
        )
    elif "profit" in cat or "w25" in cat or "wealth" in cat or "leadership" in cat or "share" in cat or "roi" in cat or "trending" in cat:
        return (
            "Acknowledge their inquiry regarding package rewards/ROI payout. "
            "Ask ONLY for: (1) Package Name (e.g., W25/W36), (2) Registered Email, and (3) The specific payout date/cycle missed."
        )
    elif "otp" in cat or "sms" in cat or "profile" in cat or "address" in cat:
        return (
            "For OTP issues, advise checking spam/junk folders and waiting 2-3 minutes before resending. "
            "Ask ONLY for: (1) Registered Email or Phone Number (with country code), and (2) Screenshot of any error."
        )
    elif "refund" in cat or "cancel" in cat or "typo" in cat or "invoice" in cat:
        return (
            "Clarify that confirmed blockchain/bank transactions cannot be reversed, though pending requests can be reviewed. "
            "Ask ONLY for: (1) Ticket/Invoice ID, (2) Exact mistake details, and (3) Payment/invoice screenshot."
        )
    else:
        return (
            "Reassure the customer that their funds are safe and our team will look into this promptly. "
            "Ask ONLY for: (1) Ticket ID, (2) Registered Email, and (3) A brief description or error screenshot."
        )

def generate_support_response(user_query):
    matches = retrieve_similar_tickets(user_query, top_k=5)
    top_category = matches[0][0]
    confidence = matches[0][3]
    
    if confidence < CONFIDENCE_THRESHOLD:
        print("\n" + "="*70)
        print(f"CUSTOMER QUERY: {user_query}")
        print(f"DETECTED SUB-SEGMENT: Unclear / Low Confidence (Match Score: {confidence:.2f})")
        print("-" * 70)
        print("SUGGESTED RESPONSE TO GIVE:")
        print("Hello! Thank you for contacting support. Could you please share a few more details about the issue you are facing (such as your Ticket ID, transaction details, or a brief description) so I can assist you accurately?")
        print("="*70 + "\n")
        return

    # Sirf subjects bhejen taake purane tickets ki amounts/IDs hallucinate na hon
    similar_subjects = "\n".join([f"- {subj} (Category: {cat})" for cat, subj, _, _ in matches[:3]])
    specific_rule = get_category_instruction(top_category)
        
    prompt = f"""Write a short, polite customer support email reply for the query below.

Customer Query: "{user_query}"
Detected Category: {top_category}
Similar Past Ticket Subjects:
{similar_subjects}

Instructions to follow strictly:
- {specific_rule}
- Do NOT invent or assume any numbers, amounts, or dates that the customer did not mention.
- Do NOT include any introductory text like "Here is the response:" or explanations after the reply. Output ONLY the message to send to the customer.
- Keep it concise (3-4 sentences plus bullet points for the requested details).
"""
    
    response = ollama.chat(
        model=LOCAL_LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.1}
    )
    
    clean_reply = response['message']['content'].strip().strip('"')
    
    print("\n" + "="*70)
    print(f"CUSTOMER QUERY: {user_query}")
    print(f"DETECTED SUB-SEGMENT: {top_category} (Match Score: {confidence:.2f})")
    print(f"LOCAL MODEL: {LOCAL_LLM_MODEL} (100% Offline & Private)")
    print("-" * 70)
    print("SUGGESTED RESPONSE TO GIVE:")
    print(clean_reply)
    print("="*70 + "\n")

if __name__ == "__main__":
    print("\n--- Local RAG Support Assistant Ready (Type 'exit' to quit) ---\n")
    while True:
        q = input("Enter customer query: ")
        if q.strip().lower() == 'exit':
            break
        if q.strip():
            generate_support_response(q)