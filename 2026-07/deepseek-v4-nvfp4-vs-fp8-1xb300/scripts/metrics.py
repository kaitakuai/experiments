"""Полный набор метрик из сырых Measurements compressa-perf 0.2.7."""
import sqlite3,sys,json
def agg(db):
    c=sqlite3.connect(db); out={}
    for eid,name in c.execute("SELECT id,experiment_name FROM Experiments"):
        rows=list(c.execute(
          "SELECT n_input,n_output,ttft,start_time,end_time,status FROM Measurements WHERE experiment_id=?",(eid,)))
        ok=[r for r in rows if (r[5] or "").lower() in ("success","ok","completed","") ]
        if not ok: ok=rows
        lat=[r[4]-r[3] for r in ok]
        ttft=[r[2] for r in ok if r[2] is not None]
        tpot=[(r[4]-r[3]-(r[2] or 0))/max(1,r[1]-1) for r in ok if r[1] and r[1]>1]
        wall=max(r[4] for r in ok)-min(r[3] for r in ok)
        nout=sum(r[1] or 0 for r in ok); nin=sum(r[0] or 0 for r in ok)
        srt=lambda a,p: sorted(a)[min(len(a)-1,int(len(a)*p))] if a else None
        out[name]={"requests":len(rows),"failed":len(rows)-len(ok),
          "TTFT":sum(ttft)/len(ttft) if ttft else None,"TTFT_95":srt(ttft,.95),
          "LATENCY":sum(lat)/len(lat),"LATENCY_95":srt(lat,.95),
          "TPOT":sum(tpot)/len(tpot) if tpot else None,
          "THROUGHPUT_OUTPUT_TOKENS":nout/wall if wall else None,
          "THROUGHPUT_INPUT_TOKENS":nin/wall if wall else None,
          "RPS":len(ok)/wall if wall else None,"wall_sec":wall}
    return out
if __name__=="__main__":
    print(json.dumps(agg(sys.argv[1]),indent=2,ensure_ascii=False))
