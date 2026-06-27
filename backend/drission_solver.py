import sys
import json
import time

def solve_and_get_cookie(token, target_url="https://chat.qwen.ai"):
    try:
        from DrissionPage import ChromiumPage, ChromiumOptions
        co = ChromiumOptions()
        co.headless(True)
        co.set_argument('--no-sandbox')
        co.set_argument('--disable-gpu')
        page = ChromiumPage(co)
        
        page.get(target_url)
        time.sleep(2)
        
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
    token = sys.argv[1] if len(sys.argv) > 1 else ""
    res = solve_and_get_cookie(token)
    print(json.dumps(res, ensure_ascii=False))
