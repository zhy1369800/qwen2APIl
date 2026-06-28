import sys
import json
import time

def solve_and_get_cookie(target_url="https://chat.qwen.ai"):
    try:
        from DrissionPage import ChromiumPage, ChromiumOptions
        co = ChromiumOptions()
        co.headless(True)
        co.set_argument('--no-sandbox')
        co.set_argument('--disable-gpu')
        page = ChromiumPage(co)
        
        # 打开具体的风控验证链接（或主页）
        page.get(target_url)
        time.sleep(3)
        
        # 提取全量 Cookie 拼装成字符串
        cookies = page.cookies()
        c_strs = []
        for c in cookies:
            name = c.get('name')
            val = c.get('value')
            if name and val:
                c_strs.append(f"{name}={val}")
        
        cookie_header = "; ".join(c_strs)
        page.quit()
        return {"success": True, "cookie": cookie_header}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # 支持命令行参数: python drission_solver.py [token] [verify_url]
    # 或者: python drission_solver.py [verify_url_or_token]
    target_url = "https://chat.qwen.ai"
    if len(sys.argv) > 2:
        target_url = sys.argv[2] if sys.argv[2].startswith("http") else sys.argv[1]
    elif len(sys.argv) > 1:
        if sys.argv[1].startswith("http"):
            target_url = sys.argv[1]
            
    res = solve_and_get_cookie(target_url)
    print(json.dumps(res, ensure_ascii=False))
