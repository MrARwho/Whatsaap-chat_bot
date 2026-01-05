import re, json, sys
from pathlib import Path

def parse_whatsapp_txt(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    # pattern: "12/4/22, 1:05 AM - Sender: message"
    msg_re = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s*([\d:]{4,8}.*?)[\u202f ]*-\s*(.*?):\s*(.*)$")
    msgs = []
    for ln in lines:
        m = msg_re.match(ln)
        if not m: 
            # continuation line: append to last message
            if msgs:
                msgs[-1]["text"] += "\n" + ln.strip()
            continue
        date, time, sender, text = m.groups()
        msgs.append({"date": date, "time": time, "sender": sender.strip(), "text": text.strip()})
    return msgs

def build_pairs(msgs, my_name="Abdul Rehman", window=1):
    pairs = []
    # For each message from someone else that is followed by a message from you, pair them.
    for i in range(len(msgs)-1):
        if msgs[i]["sender"] != my_name and msgs[i+1]["sender"] == my_name:
            prompt = msgs[i]["text"]
            response = msgs[i+1]["text"]
            pairs.append({"prompt": prompt, "response": response})
    return pairs

if __name__ == "__main__":
    inpath = sys.argv[1]   # e.g. exported_chat.txt
    outpath = sys.argv[2]  # e.g. pairs.jsonl
    my_name = sys.argv[3] if len(sys.argv)>3 else "Abdul Rehman"
    msgs = parse_whatsapp_txt(inpath)
    pairs = build_pairs(msgs, my_name=my_name)
    with open(outpath,"w",encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False)+"\n")
    print("wrote", len(pairs), "pairs to", outpath)