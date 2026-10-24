import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch

BLUE="#0072B2"; ORANGE="#E69F00"; VERM="#D55E00"; GREEN="#009E73"; SKY="#56B4E9"; PURPLE="#CC79A7"
INK="#222222"; GREY="#6B6B6B"; LIGHT="#F3F5F7"; LINE="#C9CED3"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.edgecolor":GREY,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK})
rng=np.random.default_rng(7)

def callout(ax,x,y,n,transform=None):
    c=Circle((x,y),0.011,transform=ax.transAxes if transform is None else transform,color=INK,zorder=20,clip_on=False)
    ax.add_patch(c)
    ax.text(x,y,str(n),transform=ax.transAxes if transform is None else transform,ha="center",va="center",color="white",fontsize=8,fontweight="bold",zorder=21)

def tile(fig,x,y,w,h,label,value,delta,state):
    ax=fig.add_axes([x,y,w,h]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.0,0.0),1,1,boxstyle="round,pad=0,rounding_size=0.06",fc="white",ec=LINE,lw=1))
    col={"good":BLUE,"watch":ORANGE,"alert":VERM}[state]; mark={"good":"●","watch":"▲","alert":"■"}[state]
    ax.add_patch(Rectangle((0,0),0.035,1,fc=col,ec="none"))
    ax.text(0.09,0.78,label,fontsize=8.5,color=GREY,va="center")
    ax.text(0.09,0.42,value,fontsize=19,fontweight="bold",color=INK,va="center")
    ax.text(0.09,0.12,f"{mark} {delta}",fontsize=8,color=col,va="center",fontweight="bold")
    return ax

# ---------- Fig 1 pipeline ----------
def fig_pipeline():
    fig,ax=plt.subplots(figsize=(10,3.6)); ax.set_xlim(0,10); ax.set_ylim(0,3.6); ax.axis("off")
    steps=[("1  Acquire","Public & internal\nsources (CSV, API,\nSQL extracts)"),("2  Validate","Schema, range &\nconcordance checks"),
           ("3  Clean &\nTransform","Dedupe, standardise\nunits & codes"),("4  Model","Star schema:\nfacts + dimensions,\nDAX/SQL measures"),
           ("5  Design","Wireframe, choose\nchart, set colour\n& layout"),("6  Test &\nPublish","User testing,\naccessibility check,\nrelease")]
    w=1.45; gap=0.2; x=0.1
    cols=[SKY,SKY,SKY,BLUE,GREEN,ORANGE]
    for i,(t,d) in enumerate(steps):
        ax.add_patch(FancyBboxPatch((x,1.2),w,1.9,boxstyle="round,pad=0.02,rounding_size=0.08",fc="white",ec=cols[i],lw=2))
        ax.add_patch(Rectangle((x,2.55),w,0.55,fc=cols[i],ec="none"))
        ax.text(x+w/2,2.82,t,ha="center",va="center",fontsize=8.5,fontweight="bold",color="white" if cols[i] in (BLUE,GREEN) else INK)
        ax.text(x+w/2,1.85,d,ha="center",va="center",fontsize=7.8,color=INK)
        if i<len(steps)-1:
            ax.add_patch(FancyArrowPatch((x+w+0.01,2.15),(x+w+gap-0.01,2.15),arrowstyle="-|>",mutation_scale=12,color=GREY))
        x+=w+gap
    ax.annotate("",xy=(9.7,0.7),xytext=(0.3,0.7),arrowprops=dict(arrowstyle="<|-",color=GREY,lw=1.2,linestyle="--"))
    ax.text(5,0.38,"Feedback loop: user comments, usage analytics and data-quality alerts feed back into steps 2-5",ha="center",fontsize=8.5,color=GREY,style="italic")
    ax.text(5,3.4,"From raw data to decision-ready visuals",ha="center",fontsize=11,fontweight="bold",color=INK)
    fig.savefig("../images/fig1_pipeline.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close(fig)

# ---------- Fig 2 exec dashboard ----------
def fig_exec():
    fig=plt.figure(figsize=(11,7.2),facecolor=LIGHT)
    fig.text(0.02,0.955,"Hospital Operations Overview",fontsize=15,fontweight="bold",color=INK)
    fig.text(0.02,0.925,"Illustrative data  |  Reporting period: last 12 months  |  Data last refreshed: 24 Oct 2026, 06:00",fontsize=8.5,color=GREY)
    # filter bar
    for i,t in enumerate(["Period ▾","Hospital site ▾","Department ▾","Patient group ▾"]):
        ax=fig.add_axes([0.62+i*0.095,0.935,0.088,0.035]); ax.axis("off"); ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=0.2",fc="white",ec=LINE,transform=ax.transAxes)); ax.text(0.5,0.5,t,ha="center",va="center",fontsize=7.5,color=INK)
    tiles=[("30-day readmission rate","11.8%","0.9 pts above target","alert"),("Avg length of stay","4.6 days","0.2 days below target","good"),
           ("ED median wait","47 min","approaching 60-min limit","watch"),("Bed occupancy","88%","within 80-90% band","good")]
    for i,t in enumerate(tiles): tile(fig,0.02+i*0.245,0.78,0.232,0.115,*t)
    # readmission trend
    ax=fig.add_axes([0.05,0.43,0.40,0.28],facecolor="white")
    m=np.arange(12); r=11.2+0.08*m+rng.normal(0,0.25,12); r[-3:]+=[0.2,0.5,0.8]
    ax.plot(m,r,color=BLUE,lw=2,marker="o",ms=4); ax.axhline(10.9,color=GREY,ls="--",lw=1); ax.text(0.1,10.62,"Target 10.9%",fontsize=7.5,color=GREY)
    ax.scatter(m[-1],r[-1],color=VERM,s=60,zorder=5,marker="s"); ax.annotate("Rising for 3 months",(m[-1],r[-1]),xytext=(5.6,12.5),fontsize=8,color=VERM,fontweight="bold",arrowprops=dict(arrowstyle="-",color=VERM))
    ax.set_xticks(m); ax.set_xticklabels(["Nov","Dec","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct"],fontsize=7.5); ax.set_ylim(10.3,13.2); ax.set_ylabel("% of discharges")
    ax.set_title("Readmissions have risen for three months in a row",loc="left",fontsize=9.5,fontweight="bold")
    ax.grid(axis="y",color="#E5E8EB",lw=0.8)
    # ED wait by hour
    ax=fig.add_axes([0.53,0.43,0.43,0.28],facecolor="white")
    h=np.arange(24); w=30+25*np.exp(-((h-19)**2)/18)+18*np.exp(-((h-11)**2)/12)+rng.normal(0,2,24)
    cols=[VERM if v>60 else (ORANGE if v>50 else BLUE) for v in w]
    ax.bar(h,w,color=cols,width=0.75); ax.axhline(60,color=GREY,ls="--",lw=1); ax.text(0,62,"60-min limit",fontsize=7.5,color=GREY)
    ax.set_xticks(range(0,24,3)); ax.set_xticklabels([f"{x}:00" for x in range(0,24,3)],fontsize=7.5); ax.set_ylabel("Median wait (min)")
    ax.set_title("ED waits peak between 5 pm and 9 pm",loc="left",fontsize=9.5,fontweight="bold"); ax.grid(axis="y",color="#E5E8EB",lw=0.8)
    # occupancy heatmap
    ax=fig.add_axes([0.05,0.07,0.40,0.28],facecolor="white")
    wards=["Medical","Surgical","Cardiac","Maternity","Paediatric","ICU"]; days=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    base=np.array([90,84,78,72,66,92])[:,None]; d=base+rng.normal(0,5,(6,7)); d[5,:]+=4; d=np.clip(d,50,100)
    im=ax.imshow(d,cmap="YlGnBu",vmin=50,vmax=100,aspect="auto")
    ax.set_xticks(range(7)); ax.set_xticklabels(days,fontsize=7.5); ax.set_yticks(range(6)); ax.set_yticklabels(wards,fontsize=7.5)
    for i in range(6):
        for j in range(7):
            ax.text(j,i,f"{d[i,j]:.0f}",ha="center",va="center",fontsize=7,color="white" if d[i,j]>82 else INK)
    ax.spines[:].set_visible(False); ax.set_title("ICU and Medical wards run closest to capacity",loc="left",fontsize=9.5,fontweight="bold")
    # right: top action list
    ax=fig.add_axes([0.53,0.07,0.43,0.28]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=0.03",fc="white",ec=LINE))
    ax.text(0.04,0.9,"What needs attention this week",fontsize=9.5,fontweight="bold",color=INK)
    items=[("■","Readmissions: review discharge process on Medical ward",VERM),("▲","ED: add evening triage nurse (5-9 pm)",ORANGE),("▲","ICU: above 94% on most days, review step-down beds",ORANGE),("●","Length of stay: on track, no action",BLUE)]
    for i,(mk,t,c) in enumerate(items):
        ax.text(0.05,0.72-i*0.17,mk,fontsize=10,color=c,va="center",fontweight="bold"); ax.text(0.10,0.72-i*0.17,t,fontsize=8.3,va="center",color=INK)
    ax.text(0.04,0.05,"Click any item to drill down to ward or patient-group detail",fontsize=7.5,color=GREY,style="italic")
    # callouts
    for n,(x,y) in enumerate([(0.015,0.90),(0.04,0.72),(0.525,0.72),(0.04,0.355),(0.525,0.355),(0.605,0.97)],1):
        fig.add_artist(Circle((x,y),0.0,transform=fig.transFigure))
        fig.text(x,y,str(n),fontsize=8,fontweight="bold",color="white",ha="center",va="center",bbox=dict(boxstyle="circle,pad=0.25",fc=INK,ec="none"),zorder=30)
    fig.savefig("../images/fig2_exec.png",dpi=200,bbox_inches="tight",facecolor=LIGHT); plt.close(fig)

# ---------- Fig 3 clinical quality ----------
def fig_quality():
    fig=plt.figure(figsize=(11,6.8),facecolor=LIGHT)
    fig.text(0.02,0.955,"Clinical Quality & Patient Safety",fontsize=15,fontweight="bold",color=INK)
    fig.text(0.02,0.925,"Illustrative data  |  Audience: quality leads, ward managers  |  Data last refreshed: 24 Oct 2026",fontsize=8.5,color=GREY)
    # SPC chart
    ax=fig.add_axes([0.05,0.50,0.52,0.36],facecolor="white")
    n=24; x=np.arange(1,n+1); y=np.clip(2.1+rng.normal(0,0.45,n),0.5,None); y[-5:]=[2.9,3.1,3.3,3.6,4.4]
    cl=2.1; ucl=3.5; lcl=0.7
    ax.axhline(cl,color=GREY,lw=1); ax.axhline(ucl,color=VERM,ls="--",lw=1); ax.axhline(lcl,color=VERM,ls="--",lw=1)
    ax.plot(x,y,color=BLUE,lw=1.6,marker="o",ms=4)
    out=y>ucl; ax.scatter(x[out],y[out],color=VERM,s=60,marker="s",zorder=5)
    ax.text(n+0.3,ucl,"Upper limit",fontsize=7.5,va="center",color=VERM); ax.text(n+0.3,cl,"Average",fontsize=7.5,va="center",color=GREY); ax.text(n+0.3,lcl,"Lower limit",fontsize=7.5,va="center",color=VERM)
    ax.annotate("Signal: above upper limit\n(investigate cause)",(n,y[-1]),xytext=(13,4.3),fontsize=8,color=VERM,fontweight="bold",arrowprops=dict(arrowstyle="->",color=VERM))
    ax.set_xlim(0,n+3.5); ax.set_xlabel("Week"); ax.set_ylabel("Infections per 1,000 patient-days")
    ax.set_title("Infection rate has moved above its normal range",loc="left",fontsize=9.5,fontweight="bold")
    # LOS vs benchmark
    ax=fig.add_axes([0.66,0.50,0.31,0.36],facecolor="white")
    dep=["Cardiac","Surgical","Medical","Paediatric","Maternity","Orthopaedic"]; los=np.array([5.4,4.9,6.8,3.1,2.4,5.9]); bm=np.array([5.0,5.0,5.8,3.2,2.5,5.5])
    o=np.argsort(los-bm); y_=np.arange(6)
    ax.barh(y_,(los-bm)[o],color=[VERM if v>0.5 else (ORANGE if v>0.2 else BLUE) for v in (los-bm)[o]])
    ax.set_yticks(y_); ax.set_yticklabels(np.array(dep)[o],fontsize=8); ax.axvline(0,color=INK,lw=1)
    ax.set_xlabel("Days above (+) or below (-) benchmark"); ax.set_title("Medical stays are 1 day\nlonger than benchmark",loc="left",fontsize=9.5,fontweight="bold")
    # funnel: sepsis bundle
    ax=fig.add_axes([0.05,0.07,0.40,0.31],facecolor="white")
    st=["Sepsis suspected","Lactate measured","Blood cultures taken","Antibiotics within 1 hour","Fluids given"]; v=np.array([100,93,86,71,68])
    ax.barh(range(5)[::-1],v,color=[SKY,SKY,SKY,VERM,BLUE],height=0.62)
    for i,val in enumerate(v): ax.text(val+1.5,4-i,f"{val}%",va="center",fontsize=8.5,fontweight="bold")
    ax.set_yticks(range(5)[::-1]); ax.set_yticklabels(st,fontsize=8); ax.set_xlim(0,115); ax.set_xlabel("% of patients completing step")
    ax.set_title("Largest drop-off: antibiotics within 1 hour",loc="left",fontsize=9.5,fontweight="bold")
    # bullet-style table
    ax=fig.add_axes([0.53,0.07,0.44,0.31]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0,rounding_size=0.03",fc="white",ec=LINE))
    ax.text(0.04,0.92,"Safety indicators vs target",fontsize=9.5,fontweight="bold")
    rows=[("Falls with harm","0.8","≤ 1.0","good"),("Pressure injuries","2.1","≤ 1.5","alert"),("Hand-hygiene compliance","89%","≥ 90%","watch"),("Medication errors","3.2","≤ 4.0","good")]
    ax.text(0.04,0.78,"Indicator",fontsize=7.5,color=GREY); ax.text(0.58,0.78,"Actual",fontsize=7.5,color=GREY); ax.text(0.78,0.78,"Target",fontsize=7.5,color=GREY)
    for i,(a,b,c,s) in enumerate(rows):
        col={"good":BLUE,"watch":ORANGE,"alert":VERM}[s]; mk={"good":"●","watch":"▲","alert":"■"}[s]; yy=0.62-i*0.16
        ax.text(0.04,yy,a,fontsize=8.5,va="center"); ax.text(0.58,yy,f"{mk} {b}",fontsize=8.5,va="center",color=col,fontweight="bold"); ax.text(0.78,yy,c,fontsize=8.5,va="center",color=GREY)
        ax.plot([0.04,0.96],[yy-0.075,yy-0.075],color="#E5E8EB",lw=0.8)
    for n_,(x_,y_) in enumerate([(0.015,0.88),(0.60,0.88),(0.015,0.40),(0.50,0.40)],1):
        fig.text(x_,y_,str(n_),fontsize=8,fontweight="bold",color="white",ha="center",va="center",bbox=dict(boxstyle="circle,pad=0.25",fc=INK,ec="none"),zorder=30)
    fig.savefig("../images/fig3_quality.png",dpi=200,bbox_inches="tight",facecolor=LIGHT); plt.close(fig)

# ---------- Fig 4 population health ----------
def fig_pop():
    fig=plt.figure(figsize=(11,6.8),facecolor=LIGHT)
    fig.text(0.02,0.955,"Population Health: Vaccination Coverage & Disease Burden",fontsize=15,fontweight="bold",color=INK)
    fig.text(0.02,0.925,"Illustrative data  |  Audience: public-health officers, policy makers  |  Regions shown as equal-sized tiles",fontsize=8.5,color=GREY)
    # tile map
    ax=fig.add_axes([0.04,0.12,0.40,0.74],facecolor=LIGHT); ax.set_xlim(0,6); ax.set_ylim(0,6); ax.axis("off")
    names=["A1","A2","A3","A4","","","B1","B2","B3","B4","B5","","C1","C2","C3","C4","C5","C6","","D1","D2","D3","D4","D5","","","E1","E2","E3","E4"]
    cov=rng.uniform(55,95,30)
    bins=[(0,65,VERM),(65,75,ORANGE),(75,85,SKY),(85,100,BLUE)]
    k=0
    for r in range(5):
        for c in range(6):
            nm=names[k]; v=cov[k]; k+=1
            if nm=="": continue
            col=[b[2] for b in bins if b[0]<=v<b[1]][0]
            ax.add_patch(Rectangle((c+0.04,4-r+0.04),0.92,0.92,fc=col,ec="white"))
            ax.text(c+0.5,4-r+0.62,nm,ha="center",va="center",fontsize=8,fontweight="bold",color="white" if col in (BLUE,VERM) else INK)
            ax.text(c+0.5,4-r+0.32,f"{v:.0f}%",ha="center",va="center",fontsize=7.5,color="white" if col in (BLUE,VERM) else INK)
    ax.set_title("Five regions fall below 65% coverage",loc="left",fontsize=9.5,fontweight="bold")
    # legend
    lx=0.04
    for lab,col in [("< 65%",VERM),("65-74%",ORANGE),("75-84%",SKY),("≥ 85%",BLUE)]:
        fig.add_artist(Rectangle((lx,0.075),0.018,0.025,transform=fig.transFigure,fc=col,ec="none")); fig.text(lx+0.024,0.0875,lab,fontsize=8,va="center"); lx+=0.095
    # line w/ uncertainty
    ax=fig.add_axes([0.54,0.52,0.43,0.34],facecolor="white")
    yrs=np.arange(2019,2027); a=np.array([88,86,82,80,81,83,84,85]); b=np.array([84,80,74,70,69,71,72,73])
    ax.fill_between(yrs,a-2,a+2,color=BLUE,alpha=0.2); ax.plot(yrs,a,color=BLUE,lw=2,marker="o",ms=4,label="Urban")
    ax.fill_between(yrs,b-3,b+3,color=VERM,alpha=0.2); ax.plot(yrs,b,color=VERM,lw=2,marker="s",ms=4,ls="--",label="Rural")
    ax.set_ylim(60,95); ax.set_ylabel("Coverage (%)"); ax.legend(frameon=False,fontsize=8,loc="lower right"); ax.grid(axis="y",color="#E5E8EB",lw=0.8)
    ax.set_title("Rural coverage fell further and remains well below urban",loc="left",fontsize=9.5,fontweight="bold")
    ax.text(2019.1,62,"Shaded band = 95% confidence interval",fontsize=7.5,color=GREY)
    # dot plot disparities
    ax=fig.add_axes([0.60,0.09,0.37,0.31],facecolor="white")
    grp=["Age 0-4","Age 5-14","Age 15-49","Age 50+","Lowest income","Highest income"]; val=[82,78,66,74,63,88]
    o=np.argsort(val); ax.hlines(range(6),50,np.array(val)[o],color=LINE,lw=2)
    ax.scatter(np.array(val)[o],range(6),s=70,color=[VERM if v<70 else BLUE for v in np.array(val)[o]],marker="o",zorder=5)
    ax.set_yticks(range(6)); ax.set_yticklabels(np.array(grp)[o],fontsize=8); ax.set_xlim(50,95); ax.set_xlabel("Coverage (%)")
    ax.set_title("Largest gaps: lowest-income households and adults 15-49",loc="left",fontsize=9.5,fontweight="bold")
    for n_,(x_,y_) in enumerate([(0.025,0.86),(0.525,0.86),(0.575,0.40)],1):
        fig.text(x_,y_,str(n_),fontsize=8,fontweight="bold",color="white",ha="center",va="center",bbox=dict(boxstyle="circle,pad=0.25",fc=INK,ec="none"),zorder=30)
    fig.savefig("../images/fig4_pop.png",dpi=200,bbox_inches="tight",facecolor=LIGHT); plt.close(fig)

# ---------- Fig 5 report one-pager wireframe ----------
def fig_report():
    fig,ax=plt.subplots(figsize=(7.2,9),facecolor="white"); ax.set_xlim(0,10); ax.set_ylim(0,13); ax.axis("off")
    ax.add_patch(Rectangle((0.3,0.3),9.4,12.4,fc="white",ec=INK,lw=1.5))
    def box(x,y,w,h,txt,fc="#EEF1F4",fs=8.5,bold=False,col=INK,ec=LINE):
        ax.add_patch(Rectangle((x,y),w,h,fc=fc,ec=ec,lw=1)); ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=col,fontweight="bold" if bold else "normal")
    box(0.6,11.7,8.8,0.7,"MONTHLY HEALTH PERFORMANCE BRIEFING  |  October 2026",fc=INK,fs=9.5,bold=True,col="white",ec=INK)
    box(0.6,10.3,8.8,1.2,"HEADLINE (one plain-language sentence)\n\"Readmissions rose for a third month; the Medical ward drives most of the increase.\"",fc="#FFF6E0",fs=9,ec=ORANGE)
    for i,(t,s) in enumerate([("Readmission rate","11.8%  ■"),("ED median wait","47 min  ▲"),("Bed occupancy","88%  ●")]):
        box(0.6+i*3.0,8.7,2.8,1.35,f"{t}\n\n{s}",fs=9)
    box(0.6,4.9,5.4,3.5,"[ CHART 1 ]\nTrend line with target\nand annotation\n(why it matters)",fs=9)
    box(6.2,4.9,3.2,3.5,"[ CHART 2 ]\nRanked bar chart:\nwards vs benchmark",fs=9)
    box(0.6,3.0,2.8,1.7,"WHAT CHANGED\n2-3 bullet points",fs=8.5,bold=True)
    box(3.6,3.0,2.8,1.7,"WHY IT MATTERS\nImpact on patients\nand cost",fs=8.5,bold=True)
    box(6.6,3.0,2.8,1.7,"RECOMMENDED ACTION\nOwner + deadline",fs=8.5,bold=True,fc="#E6F2FA",ec=BLUE)
    box(0.6,1.4,8.8,1.4,"DATA NOTES\nSource, definition of each metric, refresh date, known limitations,\nsmall counts suppressed (< 11), contact for questions",fs=8,fc="#F8F8F8")
    box(0.6,0.55,8.8,0.6,"Accessible version: tagged PDF with alt text for every chart  |  Page 1 of 1",fs=7.5,fc="white",col=GREY)
    for n_,(x_,y_) in enumerate([(0.3,12.05),(0.3,10.9),(0.3,9.4),(0.3,6.6),(0.3,3.9),(0.3,2.1)],1):
        ax.text(x_,y_,str(n_),fontsize=8,fontweight="bold",color="white",ha="center",va="center",bbox=dict(boxstyle="circle,pad=0.25",fc=INK,ec="none"),zorder=30)
    fig.savefig("../images/fig5_report.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close(fig)

# ---------- Fig 6 palette ----------
def fig_palette():
    fig,ax=plt.subplots(figsize=(10,3.3)); ax.set_xlim(0,10); ax.set_ylim(0,3.3); ax.axis("off")
    ax.text(0,3.1,"Colour-blind-safe palette with redundant shape cues",fontsize=11,fontweight="bold")
    sw=[("Blue","#0072B2","On target","●"),("Orange","#E69F00","Watch","▲"),("Vermillion","#D55E00","Needs action","■"),("Sky blue","#56B4E9","Context / comparison",""),("Green","#009E73","Positive change",""),("Dark grey","#444444","Text, axes","")]
    for i,(n,h,u,m) in enumerate(sw):
        x=i*1.65
        ax.add_patch(Rectangle((x,1.35),1.5,1.2,fc=h,ec="none"))
        ax.text(x+0.75,1.95,m,ha="center",va="center",fontsize=22,color="white" if h in("#0072B2","#D55E00","#444444","#009E73") else "#222222")
        ax.text(x,1.15,n,fontsize=8.5,fontweight="bold",va="top"); ax.text(x,0.9,h,fontsize=8,color=GREY,va="top"); ax.text(x,0.65,u,fontsize=8,va="top")
    ax.text(0,0.12,"Status is always shown by colour + shape + text label, never by colour alone.",fontsize=8.5,style="italic",color=GREY)
    fig.savefig("../images/fig6_palette.png",dpi=200,bbox_inches="tight",facecolor="white"); plt.close(fig)

for f in (fig_pipeline,fig_exec,fig_quality,fig_pop,fig_report,fig_palette): f()
print("ok")
