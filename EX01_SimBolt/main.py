# coding: utf-8
import os
import json
import traceback
import clr

# WPF
clr.AddReference("PresentationFramework")
clr.AddReference("PresentationCore")
clr.AddReference("WindowsBase")

from System.IO import FileStream, FileMode, FileAccess
from System.Windows.Markup import XamlReader
from System.Windows import Window, Application
from System.Windows import Visibility

# .NET HTTP
clr.AddReference("System")
from System import Text
from System.IO import StreamReader
from System.Net import WebRequest


PANEL_TITLE = "AI Assistant"


def _log(msg):
    try:
        ExtAPI.Log.WriteMessage("[EX01_SimBolt] " + msg)
    except:
        pass


def _this_dir():
    return os.path.dirname(__file__)


def _xaml_path():
    return os.path.join(_this_dir(), "ui", "AIAssistant.xaml")


def _config_path():
    return os.path.join(_this_dir(), "config.json")


def _load_xaml(path):
    fs = FileStream(path, FileMode.Open, FileAccess.Read)
    try:
        return XamlReader.Load(fs)
    finally:
        fs.Close()


def _append_chat(ui, text):
    chat = ui.FindName("ChatLog")
    if chat is None:
        return
    old = chat.Text or ""
    chat.Text = (old + "\r\n" + text) if old else text
    try:
        chat.ScrollToEnd()
    except:
        pass


def _set_status(ui, text):
    st = ui.FindName("StatusText")
    if st is not None:
        st.Text = text


def _safe_read_json_file(path):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    except:
        return None


def _config_is_valid(cfg):
    if not cfg:
        return False
    for k in ["base_url", "api_key", "model"]:
        if k not in cfg or not cfg[k] or not str(cfg[k]).strip():
            return False
    return True


def _load_config():
    return _safe_read_json_file(_config_path())


def _save_config(cfg):
    # 不要在日志输出 api_key
    p = _config_path()
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    _log("Config saved to: " + p)


def _get_cfg_fields(ui):
    base_url = ui.FindName("CfgBaseUrl")
    model = ui.FindName("CfgModel")
    api_key = ui.FindName("CfgApiKey")  # PasswordBox
    panel = ui.FindName("ConfigPanel")
    return base_url, model, api_key, panel


def _show_config_panel(ui, show):
    panel = ui.FindName("ConfigPanel")
    if panel is None:
        return
    panel.Visibility = Visibility.Visible if show else Visibility.Collapsed


def _http_post_json(url, headers, payload_obj, timeout_ms=60000):
    req = WebRequest.Create(url)
    req.Method = "POST"
    req.ContentType = "application/json"
    req.Timeout = timeout_ms

    if headers:
        for k, v in headers.items():
            try:
                req.Headers.Add(k, v)
            except:
                pass

    body = json.dumps(payload_obj)
    body_bytes = Text.Encoding.UTF8.GetBytes(body)
    req.ContentLength = len(body_bytes)

    stream = req.GetRequestStream()
    try:
        stream.Write(body_bytes, 0, len(body_bytes))
    finally:
        stream.Close()

    resp = req.GetResponse()
    try:
        sr = StreamReader(resp.GetResponseStream(), Text.Encoding.UTF8)
        try:
            return sr.ReadToEnd()
        finally:
            sr.Close()
    finally:
        try:
            resp.Close()
        except:
            pass


def _call_llm(user_text, cfg):
    base_url = str(cfg["base_url"]).rstrip("/")
    url = base_url + "/chat/completions"  # 如网关路径不同，只改这里

    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "user", "content": user_text}
        ],
        "stream": False
    }

    headers = {
        "Authorization": "Bearer " + cfg["api_key"]
    }

    return _http_post_json(url, headers, payload, timeout_ms=60000)


def _extract_assistant_text(resp_text):
    obj = json.loads(resp_text)
    try:
        return obj["choices"][0]["message"]["content"]
    except:
        return resp_text


def _wire_ui_events(ui):
    btn_send = ui.FindName("SendButton")
    prompt = ui.FindName("PromptBox")
    btn_save = ui.FindName("SaveConfigButton")
    btn_hide = ui.FindName("HideConfigButton")

    base_url_box, model_box, api_key_box, panel = _get_cfg_fields(ui)

    if btn_send is None or prompt is None:
        _log("SendButton/PromptBox not found in XAML.")
        return

    # Save config
    if btn_save is not None:
        def on_save(sender, e):
            try:
                base_url = (base_url_box.Text or "").strip() if base_url_box is not None else ""
                model = (model_box.Text or "").strip() if model_box is not None else ""
                api_key = (api_key_box.Password or "").strip() if api_key_box is not None else ""

                cfg = {"base_url": base_url, "model": model, "api_key": api_key}

                if not _config_is_valid(cfg):
                    _append_chat(ui, "Assistant: 配置不完整，请填写 URL / Model / API Key。")
                    _set_status(ui, "Config invalid")
                    return

                _save_config(cfg)
                _append_chat(ui, "Assistant: 配置已保存，可以开始聊天。")
                _set_status(ui, "Config saved")
                _show_config_panel(ui, False)
            except Exception as ex:
                _append_chat(ui, "ERROR saving config: " + str(ex))
                _append_chat(ui, traceback.format_exc())
                _set_status(ui, "Error")

        btn_save.Click += on_save

    # Hide config panel (does not delete)
    if btn_hide is not None:
        def on_hide(sender, e):
            _show_config_panel(ui, False)
        btn_hide.Click += on_hide

    # Send message
    def on_send(sender, e):
        try:
            user_text = (prompt.Text or "").strip()
            if not user_text:
                return

            cfg = _load_config()
            if not _config_is_valid(cfg):
                _append_chat(ui, "Assistant: 检测到未配置模型连接，请先在上方填写 URL / Model / API Key 并点击 Save。")
                _set_status(ui, "Need config")
                _show_config_panel(ui, True)
                return

            _append_chat(ui, "You: " + user_text)
            prompt.Text = ""
            _set_status(ui, "Calling model...")

            resp_text = _call_llm(user_text, cfg)
            assistant = _extract_assistant_text(resp_text)

            _append_chat(ui, "Assistant: " + assistant)
            _set_status(ui, "Done")
        except Exception as ex:
            _append_chat(ui, "ERROR: " + str(ex))
            _append_chat(ui, traceback.format_exc())
            _set_status(ui, "Error")
            _log("Send failed: {0}".format(ex))
            _log(traceback.format_exc())

    btn_send.Click += on_send
    _log("UI events wired.")


def _init_first_run_ui_state(ui):
    cfg = _load_config()
    if _config_is_valid(cfg):
        _show_config_panel(ui, False)
        _append_chat(ui, "Assistant: 配置已加载，可以开始聊天。")
        _set_status(ui, "Ready")
    else:
        _show_config_panel(ui, True)
        _append_chat(ui, "Assistant: 首次使用，请先配置 URL / Model / API Key，然后点击 Save。")
        _set_status(ui, "Need config")


def _show_floating_window():
    xaml = _xaml_path()
    if not os.path.exists(xaml):
        _log("ERROR: XAML not found: " + xaml)
        return

    ui = _load_xaml(xaml)
    _wire_ui_events(ui)
    _init_first_run_ui_state(ui)

    win = Window()
    win.Title = PANEL_TITLE
    win.Width = 480
    win.Height = 720
    win.Content = ui
    win.ShowInTaskbar = False

    win.Show()
    win.Activate()
    _log("Floating window shown.")


def StartUp(*args, **kwargs):
    _log("StartUp called.")
    try:
        app = Application.Current
        if app is not None and app.Dispatcher is not None:
            app.Dispatcher.BeginInvoke(_show_floating_window)
        else:
            _show_floating_window()
    except Exception as e:
        _log("ERROR in StartUp: {0}".format(e))
        _log(traceback.format_exc())
