class Kiosk:
    def __init__(self):
        # 메뉴와 가격 초기화
        self.menu = ['americano', 'latte', 'mocha', 'yuza_tea', 'green_tea', 'choco_latte']
        self.price = [2000, 3000, 3000, 2500, 2500, 3000]

    # 메뉴 출력 메서드
    def menu_print(self):
        for i in range(len(self.menu)):
            print(i + 1, self.menu[i], ' : ', self.price[i], '원')

    # 주문 메서드
    def menu_select(self):
        print()  # 줄 바꿈

        # 주문 리스트와 가격 리스트 초기화
        self.order_menu = []  # 주문 리스트
        self.order_price = []  # 가격 리스트

        n = 1  # 반복문 진입을 위해 0이 아닌 값으로 초기화

        # 0이 아닐 동안 반복 (추가 주문 로직)
        while n != 0:
            self.menu_print()  # 메뉴판 출력
            print()  # 가독성을 위한 줄 바꿈

            n = int(input("음료 번호를 입력하세요 (주문 종료는 0) : "))  # 음료 번호 입력

            # 0을 입력하면 주문 종료
            if n == 0:
                print("주문이 완료되었습니다.")
                break  # while 루프 탈출

            # 메뉴판에 있는 음료 번호일 때 (1 ~ 메뉴개수)
            if 1 <= n <= len(self.menu):
                # 음료 온도 물어보기
                t = 0  # 기본값
                while t != 1 and t != 2:  # 1이나 2를 입력할 때까지 물어보기
                    t = int(input("HOT 음료는 1을, ICE 음료는 2를 입력하세요 : "))

                    if t == 1:
                        self.temp = "HOT"
                    elif t == 2:
                        self.temp = "ICE"
                    else:
                        print("1과 2 중 하나를 입력하세요.\n")

                # 주문 내역 리스트에 추가
                self.order_price.append(self.price[n - 1])  # 가격 리스트에 추가
                self.order_menu.append(self.temp + ' ' + self.menu[n - 1])  # 주문 리스트에 추가

                # 주문 확인 메시지
                print('주문 음료', self.temp, self.menu[n - 1], ' : ', self.price[n - 1], '원')
                print("-" * 30)  # 주문 간 구분선

            # 메뉴판에 없는 번호일 때
            else:
                print("없는 메뉴입니다. 다시 주문해 주세요.")

        # 최종 주문 내역 출력
        print("=" * 30)
        print("<< 주문 내역 확인 >>")
        print("주문 리스트:", self.order_menu)
        print("가격 리스트:", self.order_price)
        print("=" * 30)

# 실행 테스트 (객체 생성 및 메서드 호출)
# kiosk = Kiosk()
# kiosk.menu_select()
# kiosk = Kiosk()
# kiosk.menu_select()